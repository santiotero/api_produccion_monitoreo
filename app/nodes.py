"""Nodos puros del grafo: supervisor, síntesis y aprobación."""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from app.providers.gemini import GeminiClient, invocar_con_reintentos
from app.schema.models import ConfigParams, SupervisorOutput
from app.state import CustomState, contribuciones_del_turno, pregunta_actual

logger = logging.getLogger("orquestador.nodes")

PROMPT_SUPERVISOR = """\
Eres el supervisor de un sistema multi-agente. Analiza las contribuciones acumuladas y decide el siguiente paso.

CONTEXTO:
- Pregunta original: {pregunta_original}
- Contribuciones: {contribuciones}
- Intentos de corrección: {intentos_correccion}
- Máximo permitido: {max_intentos_correccion}
- Aprobación humana previa: {aprobacion_previa}

REGLAS (orden de prioridad; la 1 es inquebrantable):
1. REGLA DE ORO — DECISIÓN HUMANA: si la "Aprobación humana previa" es "APROBADA" o "RECHAZADA", la decisión humana tiene TOTAL PRIORIDAD: elige next='sintesis' de INMEDIATO para informar la resolución al cliente. PROHIBIDO volver a elegir 'aprobacion'. No pases por 'investigador' ni 'analista'. (prevalece sobre todas las demás)
2. APROBACIÓN HUMANA REQUERIDA: si el cliente pide un descuento por pago anual, o un descuento motivado por insatisfacción, queja o baja del servicio → next='aprobacion' DIRECTAMENTE, sin pasar por otros agentes (prioridad sobre la regla 5).
3. Si intentos_correccion >= max_intentos_correccion → next='sintesis'.
4. Si todavía no hay contribuciones → next='investigador' con una instrucción concreta.
5. Si es una consulta con números/porcentajes/precios/descuentos → next='analista' antes que 'sintesis', salvo que en este turno ya exista una contribución del analista. La instrucción debe llevar los números exactos a calcular; una mención numérica en la contribución del investigador NO cuenta como verificación.
6. Si el analista cometió un error de cálculo o el investigador trajo datos incompletos → envía una instrucción de ajuste.
7. Si necesitas más información → next='investigador'; verificar cálculos → next='analista'.
8. Si la tarea está resuelta → next='sintesis'.
9. Contribuciones con "ERROR" o "límite de iteraciones" → next='sintesis'.

OUTPUT (Pydantic):
- next_node: Literal['investigador', 'analista', 'aprobacion', 'sintesis']
- instruccion: str (instrucción ultra-específica para el siguiente agente)
- razon: str (justificación de tu decisión)
"""

PROMPT_SINTESIS = """\
Eres el redactor final de un sistema multi-agente. Tu tarea es generar una respuesta unificada y coherente.

PREGUNTA ORIGINAL DEL USUARIO:
{pregunta_original}

CONTRIBUCIONES DE LOS AGENTES:
{contribuciones}

DECISIÓN HUMANA SOBRE EL PEDIDO:
{decision_humana}

REGLAS:
1. Sintetiza la información y responde la pregunta concreta del usuario de forma breve y directa.
2. Máximo 8 líneas. Haz respuestas directas. Si una tabla/lista excede el límite, resume en párrafos.
3. Si faltan datos o hay inconsistencias, indícalo en una línea.
4. Responde en español de forma cordial.
5. No agregues información que no esté en las contribuciones.
6. Respuesta para lectura fácil: si es necesario, usa listas y/o tablas; prioriza párrafos cortos.
"""

INSTRUCCION_FINAL = "Generar respuesta final con la información disponible."


def _formato_contribuciones(contribuciones: list[dict[str, Any]]) -> str:
    if not contribuciones:
        return "(todavía no hay contribuciones)"
    return json.dumps(contribuciones, ensure_ascii=False, indent=2, default=str)


def crear_nodo_supervisor(cliente: GeminiClient, cfg: ConfigParams) -> Any:
    """Nodo `supervisor`: función pura async que delega toda la evaluación al LLM."""
    supervisor_llm = cliente.estructurado(SupervisorOutput)

    async def nodo_supervisor(state: CustomState, config: RunnableConfig) -> dict[str, Any]:
        pregunta = pregunta_actual(state)
        contribuciones = contribuciones_del_turno(state)
        intentos = int(state.get("intentos_correccion") or 0)

        # Estado de la decisión humana: None = sin decidir, True/False = ya decidida
        approved = state.get("approved")
        if approved is None:
            aprobacion_previa = "sin decisión"
        else:
            aprobacion_previa = "APROBADA" if approved else "RECHAZADA"

        prompt = PROMPT_SUPERVISOR.format(
            pregunta_original=pregunta,
            contribuciones=_formato_contribuciones(contribuciones),
            intentos_correccion=intentos,
            max_intentos_correccion=cfg.max_intentos_correccion,
            aprobacion_previa=aprobacion_previa,
        )

        try:
            resultado = await invocar_con_reintentos(
                supervisor_llm,
                [HumanMessage(prompt)],
                intentos=2,
            )
        except Exception as exc:
            raise RuntimeError(
                f"El supervisor no pudo invocar al LLM tras reintentar: {type(exc).__name__}: {exc}"
            ) from exc

        nuevo_valor_intentos = intentos
        if resultado.next_node in ("investigador", "analista"):
            nuevo_valor_intentos += 1

        # Guarda dura: si la decisión humana ya se tomó, nunca volver a HITL
        if approved is not None and resultado.next_node == "aprobacion":
            logger.info(
                "Supervisor: se bloqueó re-solicitud de aprobación (approved=%s); se fuerza sintesis",
                approved,
            )
            resultado.next_node = "sintesis"
            resultado.instruccion = INSTRUCCION_FINAL

        if nuevo_valor_intentos >= cfg.max_intentos_correccion:
            return {
                "intentos_correccion": nuevo_valor_intentos,
                "next_agent": "sintesis",
                "instruccion_actual": INSTRUCCION_FINAL,
            }

        respuesta: dict[str, Any] = {
            "intentos_correccion": nuevo_valor_intentos,
            "next_agent": resultado.next_node,
            "instruccion_actual": resultado.instruccion,
            "requires_approval": resultado.next_node == "aprobacion",
        }
        # No pisar el motivo de aprobación que ya registró el humano
        if approved is None:
            respuesta["approval_reason"] = (
                resultado.razon if resultado.next_node == "aprobacion" else ""
            )
        return respuesta

    return nodo_supervisor


def crear_nodo_sintesis(cliente: GeminiClient, cfg: ConfigParams) -> Any:
    """Nodo `sintesis`: función pura async que redacta la respuesta unificada."""
    llm = cliente.modelo

    async def nodo_sintesis(state: CustomState, config: RunnableConfig) -> dict[str, Any]:
        pregunta = pregunta_actual(state)
        contribuciones = contribuciones_del_turno(state)

        # Dato de entrada desde el estado (mismo origen que la pregunta)
        approved = state.get("approved")
        motivo = (state.get("approval_reason") or "").strip()
        if approved is None:
            decision_humana = "sin decisión"
        elif approved:
            decision_humana = f"APROBADA. Motivo: {motivo}" if motivo else "APROBADA"
        else:
            decision_humana = f"RECHAZADA. Motivo: {motivo}" if motivo else "RECHAZADA"

        prompt = PROMPT_SINTESIS.format(
            pregunta_original=pregunta,
            contribuciones=_formato_contribuciones(contribuciones),
            decision_humana=decision_humana,
        )

        try:
            respuesta = await invocar_con_reintentos(
                llm,
                [HumanMessage(prompt)],
                intentos=2,
            )
        except Exception as exc:
            raise RuntimeError(
                f"La síntesis no pudo invocar al LLM tras reintentar: {type(exc).__name__}: {exc}"
            ) from exc

        contenido = respuesta.content if isinstance(respuesta, AIMessage) else respuesta
        return {"messages": [AIMessage(content=contenido, name="sintesis")]}

    return nodo_sintesis


def crear_nodo_aprobacion(cliente: GeminiClient, cfg: ConfigParams) -> Any:
    """Nodo `aprobacion`: nodo de control HITL.

    No invoca al LLM. Siempre retorna `next_agent: "supervisor"` para que el
    supervisor decida el siguiente paso tras la aprobación humana.
    """
    async def nodo_aprobacion(state: CustomState, config: RunnableConfig) -> dict[str, Any]:
        return {
            "next_agent": "supervisor",
            "approved": state.get("approved"),
            "approval_reason": state.get("approval_reason", ""),
            "approval_timestamp": state.get("approval_timestamp"),
        }

    return nodo_aprobacion

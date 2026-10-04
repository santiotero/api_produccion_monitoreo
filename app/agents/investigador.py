"""Agente investigador: `create_react_agent` con la herramienta `buscar_informacion`."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from app.agents import (
    AgenteReAct,
    crear_react_agent,
    extraer_herramientas,
    recortar_contexto,
)
from app.providers.gemini import GeminiClient, invocar_con_reintentos
from app.schema.models import ConfigParams
from app.state import CustomState, pregunta_actual, texto_plano
from app.tools import crear_buscar_informacion

SISTEMA = """\
Eres un investigador especializado. Tu tarea es buscar información relevante \
para responder la pregunta del usuario.

INSTRUCCIÓN DEL SUPERVISOR:
{instruccion_actual}

HERRAMIENTA DISPONIBLE:
- buscar_informacion(query): busca documentos en la base de conocimiento (RAG híbrido BM25 + Pinecone)

REGLAS:
1. Usa buscar_informacion con queries específicas y relevantes.
2. Si los resultados son incompletos, refina la query.
3. Retorna los datos encontrados de forma clara y estructurada.
4. No inventes datos. Si no encuentras algo, indícalo explícitamente.
5. NO resuelvas operaciones matemáticas ni entregues resultados numéricos calculados, devuelve los datos tal como aparecen en las fuentes, sin transformarlos, resumirlos ni calcular sobre ellos.
"""


def _salida_final(mensajes: list[Any]) -> str:
    for mensaje in reversed(mensajes):
        if isinstance(mensaje, AIMessage) and not getattr(mensaje, "tool_calls", None):
            texto = texto_plano(mensaje.content)
            if texto.strip():
                return texto
    if mensajes and isinstance(mensajes[-1], AIMessage):
        texto = texto_plano(mensajes[-1].content)
        if texto.strip():
            return texto + "\n[el agente alcanzó el límite de iteraciones antes de concluir]"
    return "El investigador no produjo salida."


def crear_agente_investigador(cliente: GeminiClient, cfg: ConfigParams) -> Any:
    """Nodo `investigador`: ReAct con `buscar_informacion` y `max_iterations` limitado."""
    agente: AgenteReAct = crear_react_agent(
        cliente.modelo,
        [crear_buscar_informacion(cfg)],
        cfg.max_iterations_agentes,
    )

    async def investigador(state: CustomState, config: RunnableConfig) -> dict[str, Any]:
        pregunta = pregunta_actual(state)
        instruccion = (
            state.get("instruccion_actual")
            or "Busca información relevante para responder la pregunta del usuario."
        )
        historial = list(state.get("messages") or [])
        mensajes = recortar_contexto(
            SISTEMA.format(instruccion_actual=instruccion),
            historial,
            cfg.max_tokens_contexto,
            pregunta,
        )

        try:
            salida = await invocar_con_reintentos(agente, mensajes, intentos=2)
        except Exception as exc:
            return {
                "contribuciones": [
                    {
                        "agente": "investigador",
                        "tipo": "datos",
                        "contenido": (
                            f"ERROR al ejecutar el investigador: {type(exc).__name__}: {exc}"
                        ),
                        "metadata": {
                            "pregunta": pregunta,
                            "instruccion": instruccion,
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                    }
                ]
            }

        mensajes_nuevos = list(salida.get("messages") or [])[len(mensajes):]
        acciones, observaciones = extraer_herramientas(mensajes_nuevos)
        contenido = _salida_final(mensajes_nuevos)

        return {
            "contribuciones": [
                {
                    "agente": "investigador",
                    "tipo": "datos",
                    "contenido": contenido,
                    "metadata": {
                        "pregunta": pregunta,
                        "instruccion": instruccion,
                        "herramientas": acciones,
                        "observaciones": observaciones,
                    },
                }
            ]
        }

    return investigador

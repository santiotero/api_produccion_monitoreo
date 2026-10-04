"""Agente analista: `create_react_agent` con la herramienta `calculadora`."""

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
from app.tools import calculadora

SISTEMA = """\
Eres un analista especializado en cálculos matemáticos exactos. Tu tarea es \
procesar datos y realizar cálculos precisos.

INSTRUCCIÓN DEL SUPERVISOR:
{instruccion_actual}

HERRAMIENTA DISPONIBLE:
- calculadora(expresion): evalúa expresiones matemáticas de forma segura (+, -, *, /, **, paréntesis)

REGLAS:
1. Usa calculadora para TODA operación matemática. No calcules mentalmente.
2. Si los datos están incompletos, indica qué falta.
3. Retorna el resultado con su contexto (unidades, significado).
4. Verifica que la expresión sea válida antes de enviarla.
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
    return "El analista no produjo salida."


def crear_agente_analista(cliente: GeminiClient, cfg: ConfigParams) -> Any:
    """Nodo `analista`: ReAct con `calculadora` y `max_iterations` limitado."""
    agente: AgenteReAct = crear_react_agent(
        cliente.modelo,
        [calculadora],
        cfg.max_iterations_agentes,
    )

    async def analista(state: CustomState, config: RunnableConfig) -> dict[str, Any]:
        pregunta = pregunta_actual(state)
        instruccion = (
            state.get("instruccion_actual")
            or "Procesa los datos disponibles y calcula el resultado exacto."
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
                        "agente": "analista",
                        "tipo": "calculo",
                        "contenido": (
                            f"ERROR al ejecutar el analista: {type(exc).__name__}: {exc}"
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
                    "agente": "analista",
                    "tipo": "calculo",
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

    return analista

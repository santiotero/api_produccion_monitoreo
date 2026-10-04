"""Agentes especialistas (ReAct) del orquestador."""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent

from app.state import texto_plano


@dataclass
class AgenteReAct:
    """Agente ReAct con control de iteraciones del bucle interno."""

    agente: CompiledStateGraph
    recursion_limit: int = 0

    async def ainvoke(
        self, messages: Sequence[BaseMessage], config: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        salida_config: dict[str, Any] = dict(config or {})
        if self.recursion_limit > 0:
            salida_config["recursion_limit"] = self.recursion_limit
        return await self.agente.ainvoke(
            {"messages": list(messages)},
            config=salida_config or None,
        )


def crear_react_agent(
    modelo: Any,
    herramientas: Sequence[Any],
    max_iterations: int,
) -> AgenteReAct:
    """Crea un agente ReAct limitado a `max_iterations` rondas de herramientas."""
    parametros = inspect.signature(create_react_agent).parameters
    if "max_iterations" in parametros:
        agente = create_react_agent(modelo, tools=list(herramientas), max_iterations=max_iterations)
        return AgenteReAct(agente=agente, recursion_limit=0)
    agente = create_react_agent(modelo, tools=list(herramientas))
    return AgenteReAct(agente=agente, recursion_limit=2 * max_iterations + 1)


def recortar_contexto(
    system_prompt: str,
    historial: Sequence[BaseMessage],
    max_tokens: int,
    pregunta: str,
) -> list[BaseMessage]:
    """System prompt + historial truncado con `trim_messages`."""
    if not historial:
        return [SystemMessage(system_prompt)]

    recortado = trim_messages(
        [SystemMessage(system_prompt), *historial],
        max_tokens=max_tokens,
        token_counter="approximate",
        strategy="last",
        start_on="human",
        include_system=True,
        allow_partial=False,
    )

    if not any(isinstance(mensaje, HumanMessage) for mensaje in recortado):
        posicion = 1 if recortado and isinstance(recortado[0], SystemMessage) else 0
        recortado.insert(posicion, HumanMessage(pregunta))
    return list(recortado)


def extraer_herramientas(
    mensajes: Sequence[BaseMessage],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """De la salida del agente extrae acciones (tool calls) y observaciones."""
    acciones: list[dict[str, Any]] = []
    observaciones: list[dict[str, Any]] = []
    for mensaje in mensajes:
        for llamada in getattr(mensaje, "tool_calls", None) or []:
            acciones.append({"herramienta": llamada.get("name"), "argumentos": llamada.get("args")})
        if getattr(mensaje, "type", None) == "tool":
            observaciones.append(
                {
                    "herramienta": getattr(mensaje, "name", None),
                    "resultado": texto_plano(getattr(mensaje, "content", "")),
                }
            )
    return acciones, observaciones

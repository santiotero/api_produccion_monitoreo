"""Cliente LLM de Gemini y helper de reintentos de API."""

from __future__ import annotations

import asyncio
import os
from typing import Any, TypeVar

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

from app.schema.models import ConfigParams

T = TypeVar("T")


class GeminiClient:
    """Cliente LLM de Gemini.

    Expone el modelo en `self.modelo` y el modelo con salida estructurada en
    `self.estructurado(esquema)`.
    """

    def __init__(self, cfg: ConfigParams) -> None:
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("Falta GEMINI_API_KEY: completa config/.env con tu key de Gemini.")
        self.cfg = cfg
        self.modelo: ChatGoogleGenerativeAI = ChatGoogleGenerativeAI(
            model=cfg.model_name,
            temperature=cfg.temperature,
            max_output_tokens=cfg.max_tokens,
            api_key=api_key,
        )

    def estructurado(self, esquema: type[BaseModel]) -> Any:
        """Runnable que invoca al LLM y valida la salida con un modelo Pydantic."""
        return self.modelo.with_structured_output(esquema)


async def invocar_con_reintentos(
    runnable: Any,
    entrada: Any,
    *,
    intentos: int = 2,
    espera: float = 1.0,
    config: dict[str, Any] | None = None,
) -> T:
    """Invoca un Runnable async reintentando ante errores transitorios de API."""
    ultimo_error: Exception | None = None
    for intento in range(1, max(intentos, 1) + 1):
        try:
            return await runnable.ainvoke(entrada, config)
        except Exception as exc:
            ultimo_error = exc
            if intento < intentos:
                await asyncio.sleep(espera * intento)
    assert ultimo_error is not None
    raise ultimo_error

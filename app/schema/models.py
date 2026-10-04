"""Esquemas Pydantic del sistema (SupervisorOutput, ConfigParams) y carga de config."""

from __future__ import annotations

import json
import os
from enum import Enum
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = BASE_DIR / "config" / "config.json"

_CLAVES_DE_RUTA = ("data_dir", "bm25_path")


class TaskStatus(str, Enum):
    """Máquina de estados de una tarea. Única fuente de verdad."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    WAITING_HUMAN = "WAITING_HUMAN"
    FAILED = "FAILED"
    DONE = "DONE"


class TaskResponse(BaseModel):
    """Respuesta de los endpoints de tareas.

    `respuesta` solo se completa cuando `status == DONE`: contiene el texto
    del nodo `sintesis`; en los demás estados es `None`.
    """

    job_id: str
    status: TaskStatus
    respuesta: Optional[str] = None


class SupervisorOutput(BaseModel):
    """Decisión estructurada que devuelve el supervisor en cada paso."""

    next_node: Literal["investigador", "analista", "aprobacion", "sintesis"] = Field(
        description="El siguiente agente a ejecutar."
    )
    instruccion: str = Field(
        description="Instrucción ultra-específica de lo que debe hacer el siguiente agente en este turno."
    )
    razon: str = Field(description="Justificación interna del supervisor para tomar esta decisión.")


class ConfigParams(BaseModel):
    """Configuración central del orquestador (config/config.json)."""

    model_name: str = "gemini-3.1-flash-lite"
    temperature: float = 0.0
    max_tokens: int = 512
    recursion_limit: int = 15
    max_iterations_agentes: int = 3
    max_tokens_contexto: int = 6000
    max_intentos_correccion: int = 3
    redis_url: str = "redis://localhost:6379"
    data_dir: str = "ingest/data"
    bm25_path: str = "app/vectorstore/bm25_index.pickle"
    embedding_model: str = "models/gemini-embedding-001"
    embedding_dim: int = 1536
    chunk_size: int = 600
    chunk_overlap: int = 100
    top_k: int = 5
    score_threshold: float = 0.75
    index_name: str = "cloudindex"
    namespace: str = "multiagente"
    phoenix_project_name: str = "api-multi-agent"
    phoenix_endpoint: str = "http://localhost:6006"

    def ruta(self, clave: str) -> Path:
        """Devuelve una ruta de configuración ya resuelta contra la raíz del proyecto."""
        valor = Path(getattr(self, clave))
        return valor if valor.is_absolute() else (BASE_DIR / valor).resolve()


def cargar_configuracion(ruta: str | Path | None = None) -> ConfigParams:
    """Lee config/config.json y devuelve un ConfigParams con las rutas resueltas."""
    path = Path(ruta) if ruta is not None else CONFIG_PATH
    if not path.is_absolute():
        path = BASE_DIR / path
    with open(path, encoding="utf-8") as fh:
        crudos = json.load(fh)
    # 12-Factor: variables de entorno (Docker) tienen prioridad sobre config.json
    crudos["redis_url"] = os.getenv("REDIS_URL", crudos.get("redis_url"))
    crudos["phoenix_endpoint"] = os.getenv("PHOENIX_ENDPOINT", crudos.get("phoenix_endpoint"))
    config = ConfigParams(**crudos)
    for clave in _CLAVES_DE_RUTA:
        setattr(config, clave, str(config.ruta(clave)))
    return config

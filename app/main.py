"""API FastAPI para el orquestador multi-agente."""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from redis.asyncio import Redis

from app.observability import configurar_observabilidad
from app.schema.models import TaskResponse, TaskStatus, cargar_configuracion

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("orquestador.api")

cfg = cargar_configuracion()
configurar_observabilidad(cfg)

redis_client: Optional[Redis] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicializa y cierra recursos de la API (solo Redis; el grafo corre en el worker)."""
    global redis_client

    redis_client = Redis.from_url(cfg.redis_url, decode_responses=True)
    logger.info("API iniciada. Redis conectado a %s", cfg.redis_url)
    yield
    await redis_client.aclose()
    logger.info("API detenida")


app = FastAPI(
    title="API Multi-Agent",
    description="API REST para orquestador multi-agente con HITL y observabilidad",
    version="1.0.0",
    docs_url="/docs",
    lifespan=lifespan,
)


class TaskRequest(BaseModel):
    pregunta: str


class ApproveRequest(BaseModel):
    approved: bool
    reason: str


@app.post("/tasks", response_model=TaskResponse, status_code=202)
async def crear_tarea(req: TaskRequest):
    """Encola tarea y devuelve job_id (HTTP 202: trabajo aceptado, aún no ejecutado)."""
    job_id = str(uuid.uuid4())
    await redis_client.hset(
        f"task:{job_id}", mapping={"status": TaskStatus.PENDING.value}
    )
    # TTL 24h: el registro de tarea se autolimpia en Redis
    await redis_client.expire(f"task:{job_id}", 86400)

    # Encolar job en ARQ
    try:
        from arq import create_pool
        from arq.connections import RedisSettings

        pool = await create_pool(RedisSettings.from_dsn(cfg.redis_url))
        await pool.enqueue_job(
            "ejecutar_tarea",
            job_id,
            req.pregunta,
            cfg.model_dump(),
            _job_id=job_id,
        )
        await pool.aclose()
    except Exception as exc:
        logger.error("Error al encolar job %s: %s", job_id, exc)
        await redis_client.hset(
            f"task:{job_id}",
            mapping={"status": TaskStatus.FAILED.value, "error": str(exc)},
        )
        raise HTTPException(status_code=500, detail="No se pudo encolar la tarea")

    logger.info("Tarea creada: job_id=%s", job_id)
    return TaskResponse(job_id=job_id, status=TaskStatus.PENDING)


@app.get("/tasks/{job_id}", response_model=TaskResponse)
async def obtener_estado(job_id: str):
    """Retorna estado actual y, si terminó, la respuesta de la síntesis."""
    estado, respuesta = await redis_client.hmget(
        f"task:{job_id}", ["status", "respuesta"]
    )
    if not estado:
        raise HTTPException(status_code=404, detail=f"Job {job_id} no encontrado")
    return TaskResponse(job_id=job_id, status=estado, respuesta=respuesta)


@app.post("/tasks/{job_id}/approve", response_model=TaskResponse, status_code=202)
async def aprobar_tarea(job_id: str, req: ApproveRequest):
    """Encola la reanudación tras aprobación humana (HTTP 202, no bloquea)."""
    estado = await redis_client.hget(f"task:{job_id}", "status")
    if not estado:
        raise HTTPException(status_code=404, detail=f"Job {job_id} no encontrado")

    if estado != TaskStatus.WAITING_HUMAN:
        raise HTTPException(
            status_code=400,
            detail=f"Job {job_id} no está esperando aprobación (estado: {estado})",
        )

    # Marcar RUNNING antes de encolar: evita doble approve concurrente
    # (el segundo verá RUNNING y recibirá 400).
    await redis_client.hset(f"task:{job_id}", mapping={"status": TaskStatus.RUNNING.value})

    try:
        from arq import create_pool
        from arq.connections import RedisSettings

        pool = await create_pool(RedisSettings.from_dsn(cfg.redis_url))
        await pool.enqueue_job(
            "reanudar_tarea",
            job_id,
            req.approved,
            req.reason,
            cfg.model_dump(),
            _job_id=f"{job_id}-approve",
        )
        await pool.aclose()
    except Exception as exc:
        logger.error("Error al encolar reanudación %s: %s", job_id, exc)
        await redis_client.hset(
            f"task:{job_id}",
            mapping={"status": TaskStatus.FAILED.value, "error": str(exc)},
        )
        raise HTTPException(status_code=500, detail="No se pudo encolar la aprobación")

    logger.info("Aprobación encolada: job_id=%s approved=%s", job_id, req.approved)
    return TaskResponse(job_id=job_id, status=TaskStatus.RUNNING)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

"""Worker ARQ para ejecución de tareas en background."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from app.graph import construir_grafo
from app.providers.gemini import GeminiClient
from app.schema.models import ConfigParams, TaskStatus
from app.state import extraer_respuesta_final

logger = logging.getLogger("orquestador.worker")


async def validar_job_executado(ctx: Any, job_id: str) -> bool:
    """Verifica si el job_id ya tiene estado final (DONE/FAILED)."""
    redis = ctx.get("redis")
    if not redis:
        return False

    estado = await redis.hget(f"task:{job_id}", "status")
    if estado in (TaskStatus.DONE, TaskStatus.FAILED):
        logger.warning("Job %s ya fue ejecutado con estado %s", job_id, estado)
        return True
    return False


async def ejecutar_tarea(
    ctx: Any,
    job_id: str,
    pregunta: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Worker que ejecuta el grafo en background.

    1. Valida idempotencia
    2. Actualiza estado a RUNNING
    3. Construye grafo con AsyncRedisSaver
    4. Ejecuta con astream
    5. Si requiere aprobación -> WAITING_HUMAN
    6. Si falla -> FAILED
    7. Si termina -> DONE
    """
    redis = ctx.get("redis")
    cfg = ConfigParams(**config)

    # 1. Validar idempotencia
    if await validar_job_executado(ctx, job_id):
        return {"job_id": job_id, "status": "ALREADY_EXECUTED"}

    # 2. Actualizar estado a RUNNING
    await redis.hset(f"task:{job_id}", mapping={"status": TaskStatus.RUNNING.value})

    # 3. Construir grafo
    from langgraph.checkpoint.redis.aio import AsyncRedisSaver

    async with AsyncRedisSaver.from_conn_string(cfg.redis_url) as saver:
        await saver.asetup()
        cliente = GeminiClient(cfg)
        grafo = construir_grafo(cliente, cfg, saver)

        # 4. Ejecutar con astream
        try:
            config_thread = {"configurable": {"thread_id": job_id}}
            estado_actual = None

            async for evento in grafo.astream(
                {"messages": [HumanMessage(content=pregunta)]},
                config=config_thread,
                stream_mode="updates",
            ):
                # interrupt_before emite la clave __interrupt__, no el nombre del nodo
                if "__interrupt__" in evento:
                    await redis.hset(
                        f"task:{job_id}",
                        mapping={"status": TaskStatus.WAITING_HUMAN.value},
                    )
                    estado_actual = TaskStatus.WAITING_HUMAN
                    logger.info("Job %s pausado esperando aprobación humana", job_id)
                    break

            # 5. Estado final: HSET único => status + respuesta atómicos
            if estado_actual != TaskStatus.WAITING_HUMAN:
                snapshot = await grafo.aget_state(config_thread)
                respuesta = extraer_respuesta_final(snapshot.values)
                await redis.hset(
                    f"task:{job_id}",
                    mapping={"status": TaskStatus.DONE.value, "respuesta": respuesta},
                )
                logger.info(
                    "Job %s completado exitosamente (%d chars de síntesis)",
                    job_id,
                    len(respuesta),
                )

            return {
                "job_id": job_id,
                "status": (estado_actual or TaskStatus.DONE).value,
            }

        except Exception as exc:
            # 6. Fallo: status limpio en el enum, detalle técnico solo en campo error
            await redis.hset(
                f"task:{job_id}",
                mapping={
                    "status": TaskStatus.FAILED.value,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            logger.error("Job %s falló: %s", job_id, exc)
            return {"job_id": job_id, "status": TaskStatus.FAILED.value}


async def reanudar_tarea(
    ctx: Any,
    job_id: str,
    approved: bool,
    reason: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Worker que reanuda el grafo tras aprobación humana.

    El endpoint POST /tasks/{id}/approve solo marca RUNNING y encola
    este job; el trabajo pesado (LLMs de supervisor/sintesis) corre aquí.
    Si quedan nodos pendientes -> WAITING_HUMAN, si no -> DONE.
    Si falla -> FAILED.
    """
    redis = ctx.get("redis")
    cfg = ConfigParams(**config)

    from langgraph.checkpoint.redis.aio import AsyncRedisSaver

    async with AsyncRedisSaver.from_conn_string(cfg.redis_url) as saver:
        await saver.asetup()
        cliente = GeminiClient(cfg)
        grafo = construir_grafo(cliente, cfg, saver)

        try:
            config_thread = {"configurable": {"thread_id": job_id}}
            await grafo.ainvoke(
                Command(
                    update={
                        "approved": approved,
                        "approval_reason": reason,
                        "approval_timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                    resume="proceder",
                ),
                config=config_thread,
            )
            snapshot = await grafo.aget_state(config_thread)
            if snapshot.next:
                await redis.hset(
                    f"task:{job_id}",
                    mapping={"status": TaskStatus.WAITING_HUMAN.value},
                )
                logger.info("Job %s volvió a pausar esperando aprobación", job_id)
                return {"job_id": job_id, "status": TaskStatus.WAITING_HUMAN.value}

            respuesta = extraer_respuesta_final(snapshot.values)
            await redis.hset(
                f"task:{job_id}",
                mapping={"status": TaskStatus.DONE.value, "respuesta": respuesta},
            )
            logger.info("Job %s reanudado a DONE (%d chars)", job_id, len(respuesta))
            return {"job_id": job_id, "status": TaskStatus.DONE.value}

        except Exception as exc:
            await redis.hset(
                f"task:{job_id}",
                mapping={
                    "status": TaskStatus.FAILED.value,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            logger.error("Job %s falló al reanudar: %s", job_id, exc)
            return {"job_id": job_id, "status": TaskStatus.FAILED.value}


async def startup_ctx(ctx: Any) -> None:
    """Inicializa el contexto del worker (Redis + observabilidad)."""
    from redis.asyncio import Redis

    from app.observability import configurar_observabilidad
    from app.schema.models import cargar_configuracion

    # Instrumentar el proceso del worker: aquí corre el grafo y los agentes
    configurar_observabilidad(cargar_configuracion())

    ctx["redis"] = Redis.from_url(
        os.getenv("REDIS_URL", "redis://localhost:6379"),
        decode_responses=True,
    )


async def shutdown_ctx(ctx: Any) -> None:
    """Cierra el contexto del worker."""
    redis = ctx.get("redis")
    if redis:
        await redis.aclose()


class WorkerSettings:
    """Configuración del worker ARQ."""

    from arq.connections import RedisSettings

    functions = [ejecutar_tarea, reanudar_tarea]
    on_startup = startup_ctx
    on_shutdown = shutdown_ctx
    redis_settings = RedisSettings.from_dsn(
        os.getenv("REDIS_URL", "redis://localhost:6379")
    )

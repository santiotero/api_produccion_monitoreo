#!/usr/bin/env python3
"""Script de prueba de carga: 5 peticiones concurrentes con casos mixtos (con o sin aprobacion).
Polling por fases en lote (batch polling) con timeout global 5 min.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

# Cargar variables de entorno (PHOENIX_API_KEY, etc.)
dotenv_path = Path(__file__).resolve().parent.parent / "config" / ".env"
load_dotenv(dotenv_path)

API_BASE = "http://localhost:8000"
PHOENIX_BASE = "http://localhost:6006"
PHOENIX_API_KEY = os.getenv("PHOENIX_API_KEY")

if not PHOENIX_API_KEY:
    print("PHOENIX_API_KEY no encontrada en config/.env")
    sys.exit(1)

TIMEOUT_FASE_2 = 300
TIMEOUT_FASE_4 = 300
SLEEP_INTERVAL = 2

CASES = [
    {
        "id": 1,
        "nombre": "Sin aprobacion (calculo)",
        "pregunta": "Dos instancias c7g.xlarge ($0,1450/h) encendidas todo el mes. Cuanto cuestan al mes?",
        "requiere_aprobacion": False,
    },
    {
        "id": 2,
        "nombre": "Aprobacion -> aprobada",
        "pregunta": "Cliente pregunta: me hacen 20% de descuento si pago anual para no darse de baja del servicio?",
        "requiere_aprobacion": True,
        "approved": True,
        "reason": "Se aprueba descuento (prueba de carga)",
    },
    {
        "id": 3,
        "nombre": "Aprobacion -> aprobada",
        "pregunta": "quiero un descuento del 100% o me doy de baja, el servicio es malo y son unos estafadores.",
        "requiere_aprobacion": True,
        "approved": True,
        "reason": "Se aprueba descuento VIP (prueba de carga)",
    },
    {
        "id": 4,
        "nombre": "Aprobacion -> rechazada",
        "pregunta": "No estoy conforme con el servicio, quiero un descuento o me doy de baja del servicio",
        "requiere_aprobacion": True,
        "approved": False,
        "reason": "Se rechaza solicitud de descuento (prueba de carga)",
    },
    {
        "id": 5,
        "nombre": "Aprobacion -> rechazada",
        "pregunta": "quiero un descuento del 30% o me doy de baja, el servicio es malo y son unos estafadores.",
        "requiere_aprobacion": True,
        "approved": False,
        "reason": "Se rechaza solicitud de descuento del 30% (prueba de carga)",
    },
]


# ---------------------------------------------------------------------------
# Helpers HTTP
# ---------------------------------------------------------------------------
async def crear_tarea(client: httpx.AsyncClient, pregunta: str) -> dict:
    """POST /tasks -> devuelve {job_id, status}."""
    resp = await client.post(f"{API_BASE}/tasks", json={"pregunta": pregunta})
    resp.raise_for_status()
    return resp.json()


async def aprobar_tarea(
    client: httpx.AsyncClient, job_id: str, approved: bool, reason: str
) -> dict:
    """POST /tasks/{job_id}/approve -> encola reanudación (202 RUNNING, no bloquea)."""
    resp = await client.post(
        f"{API_BASE}/tasks/{job_id}/approve",
        json={"approved": approved, "reason": reason},
    )
    resp.raise_for_status()
    return resp.json()


async def obtener_estado(client: httpx.AsyncClient, job_id: str) -> dict:
    """GET /tasks/{job_id} -> {status, respuesta?}."""
    resp = await client.get(f"{API_BASE}/tasks/{job_id}")
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Helpers de polling por fases
# ---------------------------------------------------------------------------
async def wait_all_status(
    client: httpx.AsyncClient,
    job_ids: list[str],
    cases: list[dict],
    target_statuses: set[str],
    timeout: int = 300,
    interval: int = 2,
) -> list[str]:
    """Poll TODOS los jobs cada 'interval' segundos hasta que TODOS los que requieren
    el target_status lo hayan alcanzado (o ya esten en DONE/FAILED).

    Args:
        client: cliente httpx compartido
        job_ids: lista de job_ids a monitorear
        cases: lista de casos (para saber cuales requieren aprobacion)
        target_statuses: conjunto de estados objetivo (ej: {"WAITING_HUMAN"})
        timeout: timeout global en segundos (default 300 = 5 min)
        interval: segundos entre polls (default 2)

    Returns:
        Lista de status finales en mismo orden que job_ids

    Raises:
        TimeoutError: si no todas llegan a target_statuses en 'timeout' segundos
    """
    start = time.time()
    print(f"  Esperando estados {target_statuses} (timeout {timeout}s, poll cada {interval}s)...")

    while time.time() - start < timeout:
        # Chequear TODOS los jobs en paralelo
        tasks = [obtener_estado(client, jid) for jid in job_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        statuses = []
        all_in_target = True
        for i, (result, case) in enumerate(zip(results, CASES)):
            if isinstance(result, Exception):
                print(f"  Job {job_ids[i][:8]}: error consultando estado: {result}")
                all_in_target = False
                break
            status = result.get("status", "")
            statuses.append(status)

            # Un job esta "completo para esta fase" si:
            # - Ya esta en target_statuses (ej: WAITING_HUMAN)
            # - O ya termino (DONE/FAILED) - esto pasa con jobs sin aprobacion
            if status not in target_statuses and status not in ("DONE", "FAILED"):
                all_in_target = False

        if all_in_target:
            print(f"  Todas en {target_statuses} (o ya finalizadas) OK")
            return statuses

        elapsed = int(time.time() - start)
        print(f"  Estados actuales: {statuses} (esperando... {elapsed}s/{timeout}s)")
        await asyncio.sleep(SLEEP_INTERVAL)

    raise TimeoutError(
        f"No todas las tareas llegaron a {target_statuses} en {timeout} segundos"
    )


async def aprobar_tareas_en_lote(
    client: httpx.AsyncClient,
    job_ids: list[str],
    cases: list[dict],
    current_statuses: list[str],
) -> None:
    """Encola reanudación SOLO para tasks en WAITING_HUMAN (202 inmediato, sin bloquear)."""
    approve_tasks = []
    for i, (case, job_id, status) in enumerate(zip(CASES, job_ids, current_statuses)):
        if case["requiere_aprobacion"] and status == "WAITING_HUMAN":
            print(f"  Aprobando job {job_id[:8]} (Caso {case['id']})...")
            approve_tasks.append(
                aprobar_tarea(client, job_id, case["approved"], case["reason"])
            )

    if approve_tasks:
        print(f"  Encolando {len(approve_tasks)} aprobaciones (202 RUNNING)...")
        resultados = await asyncio.gather(*approve_tasks, return_exceptions=True)
        for case, res in zip(
            [c for c, s in zip(CASES, current_statuses) if c["requiere_aprobacion"] and s == "WAITING_HUMAN"],
            resultados,
        ):
            if isinstance(res, Exception):
                print(f"  Caso {case['id']}: error al aprobar: {res}")
    else:
        print("  No hay aprobaciones requeridas en este momento")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main() -> None:
    # Pre-flight: salud API
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{API_BASE}/health")
        if resp.status_code != 200 or resp.json().get("status") != "healthy":
            print("La API no esta saludable (check docker compose up -d). Abortando.")
            return

        # Pregunta interactiva: limpiar Phoenix?
        try:
            ans = input(
                "Eliminar el proyecto 'api-multi-agent' de Phoenix para arrancar limpio? [s/N]: "
            )
        except EOFError:
            ans = "n"
            print("   (Usando valor por defecto: no limpiar)")
        limpiar = ans.lower() == "s"

        if limpiar and PHOENIX_API_KEY:
            print("Borrando proyecto api-multi-agent de Phoenix...")
            await client.delete(
                f"{PHOENIX_BASE}/v1/projects/api-multi-agent",
                headers={"Authorization": f"Bearer {PHOENIX_API_KEY}"},
            )
            print("   Proyecto eliminado (se recreara solo con la primera traza).")

        # FASE 1: Crear 5 tareas concurrentes
        print("\nFASE 1: Creando 5 tareas concurrentes...")
        create_tasks = [crear_tarea(client, case["pregunta"]) for case in CASES]
        creados = await asyncio.gather(*create_tasks)
        job_ids = [c["job_id"] for c in creados]

        for case, creado in zip(CASES, creados):
            print(f"  Caso {case['id']} ({case['nombre']}): job_id={creado['job_id']}")

        # FASE 2: Esperar WAITING_HUMAN solo para jobs que requieren aprobacion
        print("\nFASE 2: Esperando WAITING_HUMAN en jobs con aprobacion (timeout 300s)...")
        try:
            statuses_fase2 = await wait_all_status(
                client, job_ids, CASES, {"WAITING_HUMAN"}, timeout=TIMEOUT_FASE_2, interval=SLEEP_INTERVAL
            )
        except TimeoutError as e:
            print(f"  TIMEOUT en FASE 2: {e}")
            final_check = await asyncio.gather(*[obtener_estado(client, jid) for jid in job_ids])
            for case, jid, st in zip(CASES, job_ids, [r.get("status", "") for r in final_check]):
                print(f"  Caso {case['id']}: {st}")
            return

        # FASE 3: Aprobar SOLO las que estan en WAITING_HUMAN y requieren aprobacion
        print("\nFASE 3: Encolando aprobaciones (202 RUNNING)...")
        await aprobar_tareas_en_lote(client, job_ids, CASES, statuses_fase2)

        # FASE 4: Esperar DONE/FAILED (batch polling, timeout 300s)
        print("\nFASE 4: Esperando DONE/FAILED (timeout 300s)...")
        try:
            statuses_finales = await wait_all_status(
                client, job_ids, CASES, {"DONE", "FAILED"}, timeout=TIMEOUT_FASE_4, interval=SLEEP_INTERVAL
            )
        except TimeoutError as e:
            print(f"  TIMEOUT en FASE 4: {e}")
            final_check = await asyncio.gather(*[obtener_estado(client, jid) for jid in job_ids])
            for case, jid, st in zip(CASES, job_ids, [r.get("status", "") for r in final_check]):
                print(f"  Caso {case['id']}: {st}")
            return

        # REPORTE FINAL
        print("\n" + "=" * 50)
        print("REPORTE DE PRUEBA DE CARGA")
        print("=" * 50)

        for case, job_id, status in zip(CASES, job_ids, statuses_finales):
            # Determinar si se registro aprobacion (status DONE y caso requiere aprobacion)
            aprobado = case["requiere_aprobacion"] and case["approved"] and status == "DONE"
            if aprobado:
                print(f"[OK] Caso {case['id']} ({case['nombre']}): DONE (aprobado)")
            elif case["requiere_aprobacion"] and status == "DONE":
                print(f"[OK] Caso {case['id']} ({case['nombre']}): DONE (rechazado)")
            else:
                print(f"[OK] Caso {case['id']} ({case['nombre']}): {status}")

        # Summary
        done = sum(1 for s in statuses_finales if s == "DONE")
        failed = sum(1 for s in statuses_finales if s == "FAILED")
        print(f"\n[DONE] Resultado: {done}/5 DONE | {failed}/5 FAILED")

        if failed > 0:
            print("Algunos casos finalizaron con error. Revisar logs.")

        # Instruccion final de capturas
        print("\nYa podes capturar las screenshots en http://localhost:6006")
        print("-> Dashboards -> proyecto 'api-multi-agent'")
        print("-> Las guardas en /screenshots/ segun tu criterio.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario.")
        sys.exit(1)
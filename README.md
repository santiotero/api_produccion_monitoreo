# API Multi-Agent con FastAPI, Redis, Arize Phoenix y ARQ

API REST asíncrona para orquestador multi-agente con intervención humana (HITL) y observabilidad.

## Características

- **FastAPI**: API REST asíncrona con documentación Swagger automática
- **LangGraph**: Grafo de agentes (supervisor, investigador, analista, síntesis, aprobación)
- **Redis**: Persistencia de estado con `AsyncRedisSaver` + cola ARQ
- **Arize Phoenix**: Observabilidad con OpenInference (trazas automáticas)
- **HITL**: Intervención humana obligatoria con `interrupt_before`
- **RAG Híbrido**: Búsqueda semántica (Pinecone) + léxica (BM25)

## Requisitos

- Python 3.12+
- Docker + Docker Compose (recomendado)
- API keys: Gemini, Pinecone, Arize Phoenix

## Instalación

### Opción 1: Docker (recomendado)

```bash
# 1. Configurar variables de entorno
cp config/.env.example config/.env
# Editar config/.env con tus API keys

# 2. Levantar servicios
docker-compose up -d

# 3. Ingestar datos (primera vez, ver "Ingesta de datos")
docker-compose exec api python -m ingest.ingest
```

### Opción 2: Nativo

```bash
# 1. Crear entorno virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Configurar variables de entorno
cp config/.env.example config/.env
# Editar config/.env con tus API keys

# 4. Ingestar datos (primera vez, ver "Ingesta de datos")
python -m ingest.ingest

# 5. Levantar Redis (requerido)
redis-server

# 6. Levantar API
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 7. Levantar worker (otra terminal)
arq app.worker.WorkerSettings
```

## Ingesta de datos

Carga los PDFs de ejemplo de `ingest/data/` al índice de Pinecone y al índice
BM25:

```bash
python -m ingest.ingest                        # nativo
docker-compose exec api python -m ingest.ingest   # Docker
```

El índice de Pinecone se crea solo si no existe, con esta configuración:

| Parámetro | Valor | Dónde se define |
|-----------|-------|-----------------|
| Índice | `cloudindex` | `config/config.json` |
| Namespace | `multiagente` | `config/config.json` |
| Dimensión | 1536 | `config/config.json` (`embedding_dim`) |
| Métrica | `cosine` | **hardcodeado** en `rag_system.py:47` |
| Cloud / Región | AWS `us-east-1` (Serverless) | **hardcodeado** en `rag_system.py:48` |
| Embeddings | `models/gemini-embedding-001` | `config/config.json` |
| Chunk size / overlap | 600 / 100 tokens | `config/config.json` |
| Top-K | 5 | `config/config.json` |
| Umbral de similitud | 0.75 | `config/config.json` (`score_threshold`) |

> La métrica y la región están fijadas en el código: Pinecone crea el índice
> en `us-east-1`. Si tu cuenta está en otra región, cambialas en
> `app/vectorstore/rag_system.py:47-48`.

Para reingerir hay que borrar antes el namespace y el pickle de BM25
(`delete_all()` en `app/vectorstore/rag_system.py`).

## Uso

### Endpoints

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| POST | `/tasks` | Encola tarea y devuelve `job_id` |
| GET | `/tasks/{job_id}` | Retorna estado actual (no bloquea) |
| POST | `/tasks/{job_id}/approve` | Aprueba o rechaza tarea crítica |
| GET | `/health` | Health check |
| GET | `/docs` | Documentación Swagger UI |

### Ejemplo de uso

```bash
# 1. Crear tarea (HTTP 202 Accepted: trabajo encolado, se autolimpia en 24h)
curl -X POST http://localhost:8000/tasks \
  -H "Content-Type: application/json" \
  -d '{"pregunta": "Quiero un descuento del 10% por pago anual"}'

# Respuesta: {"job_id": "abc-123", "status": "PENDING"}

# 2. Consultar estado (respuesta: null hasta que esté DONE)
curl http://localhost:8000/tasks/abc-123

# Respuesta: {"job_id": "abc-123", "status": "WAITING_HUMAN", "respuesta": null}

# 3. Aprobar tarea (HTTP 202 Accepted: reanudación encolada, no bloquea)
curl -X POST http://localhost:8000/tasks/abc-123/approve \
  -H "Content-Type: application/json" \
  -d '{"approved": true, "reason": "Se aprueba descuento del 10%"}'

# Respuesta: {"job_id": "abc-123", "status": "RUNNING", "respuesta": null}

# 4. Consultar estado final (incluye la síntesis del nodo `sintesis`)
curl http://localhost:8000/tasks/abc-123

# Respuesta: {"job_id": "abc-123", "status": "DONE", "respuesta": "El descuento aprobado..."}
```

## Flujo de Trabajo

1. **Cliente** → `POST /tasks` con pregunta
2. **API** → Encola job en ARQ, devuelve `job_id`
3. **Worker** → Ejecuta grafo, supervisor detecta solicitud de descuento
4. **Supervisor** → `next_node = "aprobacion"`, `requires_approval = True`
5. **Grafo** → Pausa en `interrupt_before=["aprobacion"]`, estado `WAITING_HUMAN`
6. **Cliente** → `GET /tasks/{job_id}` → `{"status": "WAITING_HUMAN"}`
7. **Humano** → `POST /tasks/{job_id}/approve {"approved": true, "reason": "..."}` → `202 RUNNING` (solo encola, no bloquea)
8. **Worker** → Reanuda grafo con `Command(update={...}, resume="proceder")` (job `reanudar_tarea`)
9. **nodo_aprobacion** → Retorna `{"next_agent": "supervisor"}`
10. **Supervisor** → Decide siguiente paso (sintesis, analista, investigador, o otra aprobación)
11. **sintesis** → Redacta respuesta final → END
12. **Cliente** → `GET /tasks/{job_id}` → `{"status": "DONE"}`

## Estados de Tareas

| Estado | Descripción |
|--------|-------------|
| `PENDING` | Tarea encolada, esperando ejecución |
| `RUNNING` | Grafo en ejecución |
| `WAITING_HUMAN` | Pausado en `interrupt_before`, esperando aprobación |
| `FAILED` | Error durante ejecución (capturado y guardado en Redis) |
| `DONE` | Tarea completada; `respuesta` contiene la síntesis del nodo `sintesis` |

Los estados están definidos en `app/schema/models.py` como `TaskStatus(str, Enum)`.
Cada tarea vive en un hash `task:{job_id}` de Redis con **TTL de 24 horas** (se autolimpia).

### Arquitectura HITL: por qué no hay `hitl.py`

La lógica de intervención humana no se concentra en un módulo único. Cada
pieza vive junto a sus pares, según lo que cambia por la misma razón:

| Pieza | Ubicación | Agrupada con |
|-------|-----------|--------------|
| Nodo `aprobacion` | `app/nodes.py` | supervisor, sintesis |
| `interrupt_before=["aprobacion"]` | `app/graph.py` | compilación del grafo |
| `POST /tasks/{id}/approve` | `app/main.py` | resto de endpoints |
| Transición a `WAITING_HUMAN` | `app/worker.py` | resto de estados |

No hay un `hitl.py` dedicado porque ningún módulo es dueño exclusivo del
flujo de aprobación: agregar un endpoint, cambiar el transporte de la cola o
modificar un nodo no obliga a tocar los demás.

## Observabilidad

Arize Phoenix dashboard: http://localhost:6006

Las trazas de LangChain/LangGraph se envían automáticamente a Phoenix mediante OpenInference.

## Estructura del Proyecto

```
Api_produccion_con_monitoreo/
├── app/
│   ├── main.py              # FastAPI app + endpoints
│   ├── graph.py             # StateGraph (nodos, aristas, compilación)
│   ├── worker.py            # Worker ARQ (ejecución en background)
│   ├── observability.py     # Configuración Arize Phoenix + OpenInference
│   ├── state.py             # CustomState (MessagesState + campos de control)
│   ├── nodes.py             # Nodos: supervisor, sintesis, aprobacion
│   ├── agents/              # Agentes ReAct (investigador, analista)
│   ├── providers/           # Cliente Gemini
│   ├── vectorstore/         # RAG híbrido (Pinecone + BM25)
│   ├── schema/              # Esquemas Pydantic
│   └── tools.py             # Herramientas (buscar_informacion, calculadora)
├── config/
│   ├── .env                 # API keys (NO versionado)
│   ├── .env.example
│   └── config.json          # Configuración central
├── ingest/
│   ├── ingest.py            # Pipeline de ingesta
│   └── data/                # PDFs fuente
├── load_test/
│   └── load_test.py         # Prueba de carga: 5 peticiones concurrentes
├── screenshots/
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Configuración

### Variables de entorno (`config/.env`)

```
GEMINI_API_KEY=tu_gemini_api_key_aqui
PINECONE_API_KEY=tu_pinecone_api_key_aqui
PHOENIX_API_KEY=tu_phoenix_api_key_aqui
REDIS_URL=redis://localhost:6379
PHOENIX_ENDPOINT=http://localhost:6006
```

`REDIS_URL` y `PHOENIX_ENDPOINT` tienen prioridad sobre `config/config.json`
(ver `cargar_configuracion` en `app/schema/models.py`); si no se definen, se
usan los valores de ese archivo.

### Configuración (`config/config.json`)

```json
{
  "model_name": "gemini-3.1-flash-lite",
  "temperature": 0.0,
  "max_tokens": 512,
  "recursion_limit": 15,
  "max_iterations_agentes": 3,
  "max_tokens_contexto": 6000,
  "max_intentos_correccion": 3,
  "redis_url": "redis://localhost:6379",
  "data_dir": "ingest/data",
  "bm25_path": "app/vectorstore/bm25_index.pickle",
  "embedding_model": "models/gemini-embedding-001",
  "embedding_dim": 1536,
  "chunk_size": 600,
  "chunk_overlap": 100,
  "top_k": 5,
  "score_threshold": 0.75,
  "index_name": "cloudindex",
  "namespace": "multiagente",
  "phoenix_project_name": "api-multi-agent",
  "phoenix_endpoint": "http://localhost:6006"
}
```

## Licencia

MIT

---

## Prueba de Carga y Evidencia de Costos

Ejecuta 5 peticiones concurrentes contra la API:
- 1 cálculo sin aprobación (Caso 1)
- 4 con aprobación HITL: 2 aprobadas (Casos 2,3) + 2 rechazadas (Casos 4,5)

Captura métricas en Phoenix: costo por ejecución y latencia p95.

# Importante: capturas del dashboard de trazas (evidencia de observabilidad)
Las capturas muestran dos p95 distintos porque la corrida cruzó dos buckets de Phoenix y cada uno calcula el suyo. Entre las dos cubren el ciclo completo: interrupción → reanudación → síntesis.
aclaracion: "Error: GeneratorExit in astream" es la interrupción esperada, no un job FAILED

### Variables de entorno
El script carga `config/.env` por su cuenta: no hace falta exportar nada.
`PHOENIX_API_KEY` debe estar definida ahí porque se usa para limpiar el
proyecto antes de la corrida.

### Ejecución
```bash
cd load_test
python load_test.py
# Responde 's' a la limpieza de Phoenix (recomendado)
# Espera reporte final (timeout 300s por fase)
```

### Flujo (4 fases)
1. **Crear 5 tareas concurrentes** → `POST /tasks` simultáneas
2. **Esperar WAITING_HUMAN** → batch polling cada 2s, solo jobs con aprobación (timeout 5 min)
3. **Aprobar/Rechazar** → `POST /approve` (202 RUNNING inmediato, la reanudación corre en el worker)
4. **Esperar DONE/FAILED** → batch polling hasta completar (timeout 5 min)

### Demoras del LLM
La reanudación corre en el worker e invoca al LLM (Gemini), que puede tener mucha demora: por eso el approve responde 202 al instante y el script espera el DONE vía polling.

### Output
- Reporte en consola (5/5 DONE, aprobados/rechazados, tiempos)
- Capturas en `/screenshots/`:
  - Costo por ejecución (calculado por Phoenix desde los tokens de entrada y salida)
  - Latencia p95 de la corrida
  - Trazas de los nodos del grafo

### URLs y puertos

| Servicio | Nativo (desde tu máquina) | Dentro de Docker | Puerto publicado |
|----------|---------------------------|-------------------|------------------|
| API | `http://localhost:8000` | `http://api:8000` | 8000 |
| Health check | `http://localhost:8000/health` | `http://api:8000/health` | 8000 |
| Swagger UI | `http://localhost:8000/docs` | `http://api:8000/docs` | 8000 |
| Redis | `redis://localhost:6379` | `redis://redis:6379` | 6379 |
| Phoenix (UI) | `http://localhost:6006` | `http://phoenix:6006` | 6006 |
| Phoenix (OTLP) | `http://localhost:6006/v1/traces` | `http://phoenix:6006/v1/traces` | 6006 |

`PHOENIX_ENDPOINT` se deja como URL base: `app/observability.py` le agrega
`/v1/traces` por su cuenta.

**Una sola Redis cumple tres roles**: estado de los jobs (`task:{job_id}`),
checkpoints de LangGraph (`AsyncRedisSaver`) y cola de trabajos (broker de
ARQ). No hace falta una segunda instancia.

`docker-compose.yml` sobreescribe `REDIS_URL` y `PHOENIX_ENDPOINT` con los
hostnames internos de la red de Docker. **No los copies a `config/.env` si
vas a correr en modo nativo**: ahí van `localhost`.

---

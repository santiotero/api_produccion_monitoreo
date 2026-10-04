"""Configuración de observabilidad con Arize Phoenix y OpenInference."""

from __future__ import annotations

import logging

from app.schema.models import ConfigParams

logger = logging.getLogger("orquestador.observability")


def configurar_observabilidad(cfg: ConfigParams):
    """Inicializa Arize Phoenix con OpenInference para trazas automáticas."""
    try:
        from phoenix.otel import register
        from openinference.instrumentation.langchain import LangChainInstrumentor
    except ImportError as exc:
        logger.warning(
            "No se pudo configurar Arize Phoenix: %s. "
            "Instala las dependencias con: pip install arize-phoenix openinference-instrumentation-langchain",
            exc,
        )
        return None

    try:
        # Phoenix acepta OTLP/HTTP en /v1/traces (la raíz / solo sirve la UI)
        endpoint = cfg.phoenix_endpoint.rstrip("/")
        if not endpoint.endswith("/v1/traces"):
            endpoint = f"{endpoint}/v1/traces"
        tracer_provider = register(
            project_name=cfg.phoenix_project_name,
            endpoint=endpoint,
            protocol="http/protobuf",
        )
        LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
        logger.info(
            "Arize Phoenix configurado: proyecto=%s, endpoint=%s",
            cfg.phoenix_project_name,
            cfg.phoenix_endpoint,
        )
        return tracer_provider
    except Exception as exc:
        logger.error("Error al configurar Arize Phoenix: %s", exc)
        return None

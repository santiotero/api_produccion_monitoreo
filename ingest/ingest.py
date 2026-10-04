"""Pipeline de ingesta: documentos (PDF/TXT/MD/JSON) → chunks → Pinecone + BM25.

Comando: `python -m ingest.ingest`
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import tiktoken
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.schema.models import BASE_DIR, ConfigParams, cargar_configuracion
from app.vectorstore.rag_system import RAGSystem

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("orquestador.ingest")

EXTENSIONES_VALIDAS = {".pdf", ".txt", ".md", ".json"}


def cargar_documentos(data_dir: Path) -> list[Document]:
    """Carga los documentos de `data_dir` según su extensión (PDF, TXT, MD, JSON)."""
    documentos: list[Document] = []
    if not data_dir.exists():
        logger.error("Carpeta de datos no existe: %s", data_dir)
        return documentos

    for archivo in sorted(data_dir.rglob("*")):
        ext = archivo.suffix.lower()
        if ext not in EXTENSIONES_VALIDAS:
            continue

        logger.info("Cargando: %s", archivo.name)
        if ext == ".pdf":
            documentos.extend(PyPDFLoader(str(archivo)).load())
        elif ext == ".json":
            documentos.extend(_cargar_json(archivo))
        else:
            documentos.extend(TextLoader(str(archivo), encoding="utf-8").load())

    logger.info("Documentos cargados: %d", len(documentos))
    return documentos


def _cargar_json(archivo: Path) -> list[Document]:
    """JSON: cada elemento de una lista superior se vuelve un documento."""
    with open(archivo, encoding="utf-8") as fh:
        datos = json.load(fh)
    fuente = {"source": str(archivo)}
    if isinstance(datos, list):
        return [
            Document(
                page_content=json.dumps(itemo, ensure_ascii=False, indent=2), metadata=dict(fuente)
            )
            for itemo in datos
        ]
    return [Document(page_content=json.dumps(datos, ensure_ascii=False, indent=2), metadata=fuente)]


def dividir_documentos(
    documentos: list[Document], chunk_size: int, chunk_overlap: int
) -> list[Document]:
    """Divide los documentos en chunks con tiktoken (600 tokens, 100 de overlap)."""
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documentos)

    codificador = tiktoken.get_encoding("cl100k_base")
    conteos = [len(codificador.encode(chunk.page_content)) for chunk in chunks]
    logger.info(
        "Chunks generados: %d (min_tokens=%d, max_tokens=%d)",
        len(chunks),
        min(conteos) if conteos else 0,
        max(conteos) if conteos else 0,
    )
    return chunks


def main() -> None:
    """Pipeline completo de ingesta: carga → split → Pinecone → BM25 (pickle)."""
    logger.info("=" * 50)
    logger.info("Proceso de Ingesta - API Multi-Agent")
    logger.info("=" * 50)

    load_dotenv(BASE_DIR / "config" / ".env")
    cfg: ConfigParams = cargar_configuracion()

    pinecone_api_key = os.environ.get("PINECONE_API_KEY", "").strip()
    if not pinecone_api_key:
        logger.error("PINECONE_API_KEY no configurada en config/.env")
        raise SystemExit(1)

    data_dir = Path(cfg.data_dir)
    logger.info("Carpeta de datos: %s", data_dir)

    rag = RAGSystem(
        cfg,
        pinecone_api_key,
        os.environ.get("GEMINI_API_KEY", "").strip() or None,
    )

    pinecone_con_datos = rag.has_data()
    pickle_existe = rag.has_bm25()

    if pinecone_con_datos and pickle_existe:
        logger.error(
            "Ya hay datos cargados en Pinecone (namespace %s) y BM25. "
            "Para re-ingestar hay que eliminar antes el namespace y el pickle.",
            cfg.namespace,
        )
        raise SystemExit(1)

    if pinecone_con_datos:
        logger.warning("Pinecone tiene datos pero BM25 no existe: puede haber inconsistencia.")
    if pickle_existe:
        logger.warning("BM25 existe pero Pinecone no tiene datos: puede haber inconsistencia.")

    documentos = cargar_documentos(data_dir)
    if not documentos:
        logger.error("No se encontraron documentos en %s", data_dir)
        raise SystemExit(1)

    chunks = dividir_documentos(documentos, cfg.chunk_size, cfg.chunk_overlap)
    rag.add_documents(chunks)

    error = rag.validate_namespace()
    if error:
        logger.error("Validación del namespace falló: %s", error)
        raise SystemExit(1)

    logger.info(
        "Ingesta finalizada: %d fragmentos en el índice '%s' / namespace '%s'.",
        len(chunks),
        cfg.index_name,
        cfg.namespace,
    )


if __name__ == "__main__":
    main()

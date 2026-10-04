"""RAG híbrido: Pinecone (denso) + BM25 (sparse) combinados con EnsembleRetriever."""

from __future__ import annotations

import logging
import pickle
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from app.schema.models import ConfigParams

logger = logging.getLogger("orquestador.rag_system")


class ClientePinecone:
    """Conecta con Pinecone, crea el índice si falta y valida el namespace."""

    def __init__(
        self,
        api_key: str,
        index_name: str,
        namespace: str,
        embedding_dim: int = 1536,
    ) -> None:
        self.api_key = api_key
        self.index_name = index_name
        self.namespace = namespace
        self.embedding_dim = embedding_dim
        self.pc = Pinecone(api_key=api_key)
        self.index = self._obtener_indice()

    def _obtener_indice(self):
        existentes = [idx.name for idx in self.pc.list_indexes()]
        if self.index_name not in existentes:
            logger.info("Creando índice '%s'...", self.index_name)
            self.pc.create_index(
                name=self.index_name,
                dimension=self.embedding_dim,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            while not self.pc.describe_index(self.index_name).status["ready"]:
                time.sleep(1)
            logger.info("Índice '%s' creado.", self.index_name)
        return self.pc.Index(self.index_name)

    def validate_namespace(self) -> str | None:
        """None si el namespace existe y tiene datos; mensaje de error si no."""
        stats = self.index.describe_index_stats()
        if self.namespace not in stats.namespaces:
            return f"El espacio {self.namespace} aún no existe"
        if stats.namespaces[self.namespace].vector_count == 0:
            return "Este espacio no contiene datos aún"
        return None

    def has_data(self) -> bool:
        """True si el namespace tiene al menos un vector."""
        try:
            stats = self.index.describe_index_stats()
            namespace = stats.namespaces.get(self.namespace)
            return namespace is not None and namespace.vector_count > 0
        except Exception as exc:
            logger.error("Error al verificar datos en Pinecone: %s", exc)
            return False

    def delete_namespace(self) -> None:
        """Elimina el namespace completo de Pinecone."""
        try:
            self.index.delete_namespace(namespace=self.namespace)
            logger.info("Namespace '%s' eliminado de Pinecone.", self.namespace)
        except Exception as exc:
            logger.error("Error al eliminar namespace de Pinecone: %s", exc)


class RAGSystem:
    """Búsqueda híbrida: similitud de vectores (dense) + BM25 (sparse)."""

    def __init__(
        self,
        config: ConfigParams,
        api_key: str,
        gemini_api_key: str | None = None,
    ) -> None:
        self.config = config
        self.bm25_path = Path(config.bm25_path)

        self.pinecone_client = ClientePinecone(
            api_key=api_key,
            index_name=config.index_name,
            namespace=config.namespace,
            embedding_dim=config.embedding_dim,
        )
        self.index = self.pinecone_client.index

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=config.embedding_model,
            google_api_key=gemini_api_key,
            output_dimensionality=config.embedding_dim,
        )

        self.vector_store = PineconeVectorStore(
            index=self.index,
            embedding=self.embeddings,
            namespace=config.namespace,
            text_key="text",
        )
        self.dense_retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": config.top_k,
                "score_threshold": config.score_threshold,
            },
        )

        self.sparse_retriever = self._cargar_bm25()
        self.ensemble_retriever = self._construir_ensemble()

    def _cargar_bm25(self) -> BM25Retriever | None:
        """Carga BM25 desde el pickle; None si todavía no existe."""
        if self.bm25_path.exists():
            logger.info("Cargando BM25 desde %s", self.bm25_path)
            with open(self.bm25_path, "rb") as fh:
                return pickle.load(fh)
        logger.info("No se encontró BM25 persistido; se crea en la ingesta.")
        return None

    def _construir_ensemble(self) -> EnsembleRetriever:
        retrievers = [self.dense_retriever]
        weights = [1.0]
        if self.sparse_retriever is not None:
            retrievers.append(self.sparse_retriever)
            weights = [0.5, 0.5]
        return EnsembleRetriever(retrievers=retrievers, weights=weights)

    def build_retriever(self) -> EnsembleRetriever:
        """Retorna el retriever híbrido (dense + sparse con RRF)."""
        return self.ensemble_retriever

    def get_relevant_docs(self, query: str) -> list:
        """Solo los documentos cuyo score de similitud supera el umbral."""
        resultados = self.vector_store.similarity_search_with_score(query, k=self.config.top_k)
        return [doc for doc, score in resultados if score >= self.config.score_threshold]

    def validate_namespace(self) -> str | None:
        return self.pinecone_client.validate_namespace()

    def has_data(self) -> bool:
        return self.pinecone_client.has_data()

    def has_bm25(self) -> bool:
        return self.bm25_path.exists()

    def add_documents(self, chunks: Iterable[Any]) -> None:
        """Persiste los chunks en Pinecone y crea/serializa el índice BM25."""
        fragmentos = list(chunks)
        logger.info("Almacenando %d chunks en Pinecone...", len(fragmentos))
        self.vector_store.add_documents(fragmentos)
        self.save_bm25(fragmentos)

    def save_bm25(self, chunks: Iterable[Any]) -> None:
        """Fitea BM25 con los chunks y lo serializa en `bm25_path`."""
        fragmentos = list(chunks)
        bm25 = BM25Retriever.from_documents(fragmentos, k=self.config.top_k)
        self.bm25_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.bm25_path, "wb") as fh:
            pickle.dump(bm25, fh)
        logger.info("BM25 serializado en %s", self.bm25_path)
        self.sparse_retriever = bm25
        self.ensemble_retriever = self._construir_ensemble()

    def delete_all(self) -> None:
        """Borra el namespace de Pinecone y el pickle de BM25."""
        self.pinecone_client.delete_namespace()
        if self.bm25_path.exists():
            self.bm25_path.unlink()
            logger.info("Archivo pickle eliminado: %s", self.bm25_path)

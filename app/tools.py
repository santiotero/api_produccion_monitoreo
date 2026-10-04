"""Herramientas de los agentes: `buscar_informacion` (RAG) y `calculadora`."""

from __future__ import annotations

import ast
import asyncio
import math
import operator
import os
from collections.abc import Callable

from langchain_core.tools import BaseTool, tool

from app.schema.models import ConfigParams
from app.vectorstore.rag_system import RAGSystem

_CACHE_RAG: dict[str, RAGSystem] = {}

_OPERADORES_BINARIOS: dict[type, Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}

_OPERADORES_UNARIOS: dict[type, Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_LIMITE_POTENCIA = 64


def _obtener_rag(cfg: ConfigParams) -> RAGSystem:
    """Instancia única (lazy) del RAGSystem para el índice/namespace configurados."""
    clave = f"{cfg.index_name}|{cfg.namespace}"
    rag = _CACHE_RAG.get(clave)
    if rag is None:
        pinecone_key = os.environ.get("PINECONE_API_KEY", "").strip()
        if not pinecone_key:
            raise RuntimeError(
                "Falta PINECONE_API_KEY: completa config/.env con tu key de Pinecone."
            )
        gemini_key = os.environ.get("GEMINI_API_KEY", "").strip() or None
        rag = RAGSystem(cfg, pinecone_key, gemini_key)
        _CACHE_RAG[clave] = rag
    return rag


def crear_buscar_informacion(cfg: ConfigParams) -> BaseTool:
    """Crea la herramienta `buscar_informacion` cerrada sobre la configuración."""

    @tool
    async def buscar_informacion(query: str) -> str:
        """Busca documentos relevantes en la base de conocimiento del proyecto.

        Usa un retriever HÍBRIDO que combina dos estrategias y las fusiona con
        Reciprocal Rank Fusion:

        * denso: similitud de embeddings de Gemini (models/gemini-embedding-001)
          en el índice Pinecone `cloudindex`, namespace `multiagente`, filtrando
          por score_threshold 0.75;
        * disperso: coincidencia léxica BM25 sobre los mismos documentos.

        Cuándo usarla:
        * cuando necesites datos, definiciones, conceptos o ejemplos contenidos
          en los documentos cargados (PDF de Python: fundamentos, POO, módulos y
          excepciones);
        * cuando la instrucción del supervisor pida "buscar", "verificar",
          "contrastar" o "recolectar" información antes de responder.

        Cómo usarla:
        * `query` debe ser una consulta breve y específica en español, con las
          palabras clave del dato que buscas (p. ej. "clases y objetos en Python",
          "manejo de excepciones try except");
        * si el resultado viene vacío o incompleto, reformula la query con otros
          términos y vuelve a llamar (máximo un par de intentos).

        Devuelve:
        * los fragmentos más relevantes numerados, con su fuente y texto
          completo, listos para citar;
        * o un mensaje explícito si no hay coincidencias (no inventes datos a
          partir de este mensaje: indícalo).
        """
        try:
            rag = await asyncio.to_thread(_obtener_rag, cfg)
        except Exception as exc:
            return (
                "No se pudo conectar con la base de conocimiento: "
                f"{exc}. Informa este error y no inventes datos."
            )

        try:
            documentos = await asyncio.to_thread(rag.build_retriever().invoke, query)
        except Exception as exc:
            return (
                f"Error al consultar la base de conocimiento: {exc}. "
                "Reintenta con otra consulta más simple."
            )

        if not documentos:
            return (
                "Sin resultados: no hay documentos relevantes para esa consulta. "
                "Prueba con otros términos o indica que el dato no está en la base."
            )

        bloques = []
        for posicion, documento in enumerate(documentos, start=1):
            fuente = documento.metadata.get("source", "desconocida")
            texto = " ".join(str(documento.page_content).split())
            bloques.append(f"[{posicion}] fuente={fuente}\n{texto}")
        return "\n\n".join(bloques)

    return buscar_informacion


def _evaluar_nodo(nodo: ast.AST) -> float:
    """Evalúa recursivamente un nodo del AST con un conjunto cerrado de operadores."""
    if isinstance(nodo, ast.Constant) and isinstance(nodo.value, (int, float)):
        if isinstance(nodo.value, bool):
            raise TypeError("Los booleanos no son válidos en una expresión numérica")
        return float(nodo.value)

    if isinstance(nodo, ast.UnaryOp) and type(nodo.op) in _OPERADORES_UNARIOS:
        return _OPERADORES_UNARIOS[type(nodo.op)](_evaluar_nodo(nodo.operand))

    if isinstance(nodo, ast.BinOp) and type(nodo.op) in _OPERADORES_BINARIOS:
        izquierda = _evaluar_nodo(nodo.left)
        derecha = _evaluar_nodo(nodo.right)
        if isinstance(nodo.op, ast.Pow) and abs(derecha) > _LIMITE_POTENCIA:
            raise ValueError(f"Exponente fuera de rango (máximo {_LIMITE_POTENCIA})")
        return _OPERADORES_BINARIOS[type(nodo.op)](izquierda, derecha)

    raise ValueError(f"Construcción no soportada: {type(nodo).__name__}")


def _formatear(numero: float) -> str:
    if float(numero).is_integer() and abs(numero) < 1e16:
        return str(int(numero))
    return repr(numero)


@tool
def calculadora(expresion: str) -> str:
    """Evalúa una expresión matemática de forma segura y devuelve el resultado exacto.

    Soporta números (enteros y decimales), los operadores `+`, `-`, `*`, `/` y
    `**` (potencia) y paréntesis para agrupar, p. ej. `(12 + 8) * 3 / 4`.

    Cuándo usarla:
    * para TODA operación aritmética: sumas, restas, multiplicaciones,
      divisiones, porcentajes, potencias y combinaciones de ellas;
    * también para verificar un cálculo que hiciste mentalmente.

    Reglas:
    * no uses letras, funciones ni variables: solo números y los operadores
      permitidos (la evaluación es un AST seguro, sin `eval`);
    * si la expresión es inválida o está fuera de rango, la herramienta devuelve
      un mensaje de error y debes corregir la expresión, no adivinar el resultado;
    * el resultado se devuelve como string: repítelo con su unidad o contexto.

    Devuelve el resultado numérico como texto, o un mensaje `ERROR: ...`.
    """
    texto = str(expresion).strip()
    if not texto:
        return "ERROR: la expresión está vacía."

    try:
        arbol = ast.parse(texto, mode="eval")
    except SyntaxError as exc:
        return f"ERROR: expresión inválida ({exc.msg}). Revisa números, signos y paréntesis."

    try:
        resultado = _evaluar_nodo(arbol.body)
    except ZeroDivisionError:
        return "ERROR: división por cero."
    except (ValueError, TypeError, OverflowError) as exc:
        return f"ERROR: {exc}"
    except Exception as exc:
        return f"ERROR: no se pudo evaluar la expresión ({type(exc).__name__}: {exc})."

    if isinstance(resultado, float) and not math.isfinite(resultado):
        return "ERROR: el resultado no es un número finito."

    return _formatear(resultado)


def crear_tools(cfg: ConfigParams) -> list[BaseTool]:
    """Lista de herramientas del sistema: `buscar_informacion` + `calculadora`."""
    return [crear_buscar_informacion(cfg), calculadora]

"""System adapters: one small interface for any RAG system under test.

A system is anything with an ``ask(question, k)`` method that returns a
:class:`SystemResult`. The runner only ever talks to that interface, so
adding a new system means writing one adapter class, not touching the
runner or the metrics.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

# Default location of the document-qa-rag checkout. Read once at import
# time from RAG_EVAL_DOC_QA_RAG_PATH, so set the variable before
# importing rag_eval (or pass --doc-qa-rag-path on the CLI) if the
# checkout lives elsewhere; changing it later has no effect.
DEFAULT_DOC_QA_RAG_PATH = Path(
    os.environ.get(
        "RAG_EVAL_DOC_QA_RAG_PATH",
        str(
            Path.home()
            / "workspace"
            / "goals"
            / "get-hired-as-an-ai-ml-engineer"
            / "hidden_files"
            / "document-qa-rag"
        ),
    )
)


@dataclass
class Citation:
    """Where an answer claims to come from."""

    species: str | None = None
    source: str | None = None
    page: int | str | None = None
    score: float | None = None


@dataclass
class SystemResult:
    """Everything the harness needs from one question."""

    answer: str
    citations: list[Citation] = field(default_factory=list)
    abstained: bool = False
    retrieved_chunk_ids: list[int] = field(default_factory=list)

    @property
    def cited_species(self) -> str | None:
        """Species of the first citation, or None when there is none."""
        return self.citations[0].species if self.citations else None


class RagSystem(Protocol):
    """Protocol for a RAG system under evaluation."""

    name: str

    def ask(self, question: str, k: int = 5) -> SystemResult:
        """Answer ``question`` and return the answer, citations, abstention
        flag and the ranked list of retrieved chunk ids."""
        ...


class DocQARagSystem:
    """Adapter for the document-qa-rag project (TF-IDF backend).

    Imports that project's ``src.answer`` and ``src.retrieve`` modules
    from a configurable path (constructor argument, or the
    ``RAG_EVAL_DOC_QA_RAG_PATH`` environment variable). Requires that
    project's own dependencies (numpy, scikit-learn) to be installed;
    the harness itself does not depend on them.
    """

    name = "doc-qa-rag"

    def __init__(
        self,
        doc_qa_rag_path: str | Path | None = None,
        index_dir: str | Path | None = None,
    ) -> None:
        self.doc_qa_rag_path = Path(doc_qa_rag_path or DEFAULT_DOC_QA_RAG_PATH)
        self.index_dir = Path(index_dir) if index_dir else self.doc_qa_rag_path / "index"
        if str(self.doc_qa_rag_path) not in sys.path:
            sys.path.insert(0, str(self.doc_qa_rag_path))
        try:
            from src.answer import answer_question  # noqa: E402
            from src.retrieve import Retriever  # noqa: E402
        except ImportError as exc:
            raise ImportError(
                "Could not import document-qa-rag modules from "
                f"{self.doc_qa_rag_path}. Make sure the path is right and "
                "that numpy and scikit-learn are installed (see README)."
            ) from exc
        self._answer_question = answer_question
        self._retriever = Retriever(self.index_dir)
        # Chunk identity: document-qa-rag stores chunk_id 0 in every chunk's
        # metadata, so that field cannot identify a chunk. The harness
        # instead uses each chunk's position in the index list as its id,
        # and the golden dataset's relevant_chunk_ids use the same numbering.
        self._chunk_ids = {id(chunk): i for i, chunk in enumerate(self._retriever.store.chunks)}

    def ask(self, question: str, k: int = 5) -> SystemResult:
        retrieved = self._retriever.retrieve(question, k=k)
        retrieved_ids = [self._chunk_ids[id(r["chunk"])] for r in retrieved]
        # Note: answer_question retrieves the same question again
        # internally, so each question is retrieved twice. That is
        # accepted: the adapter cannot pass pre-retrieved chunks into
        # answer_question without changing document-qa-rag, and TF-IDF
        # retrieval over 744 in-memory chunks is cheap.
        result = self._answer_question(self._retriever, question, k=k)
        citations: list[Citation] = []
        citation = result.get("citation")
        if citation:
            citations.append(
                Citation(
                    species=citation.get("species"),
                    source=citation.get("source"),
                    page=citation.get("page"),
                    score=citation.get("score"),
                )
            )
        return SystemResult(
            answer=result.get("answer", ""),
            citations=citations,
            abstained=bool(result.get("abstained", False)),
            retrieved_chunk_ids=retrieved_ids,
        )


class HttpRagSystem:
    """Adapter for any HTTP RAG service with an /ask-style endpoint.

    POSTs JSON ``{"question": ...}`` and accepts a JSON reply shaped
    like ``{"answer": ..., "citations": [...], "abstained": ...}``,
    tolerating common variations (``retrieved_chunk_ids`` /
    ``retrieved_ids`` / ``chunk_ids``, citations as strings or dicts).
    Uses only the standard library.
    """

    def __init__(self, endpoint: str, timeout: float = 30.0) -> None:
        if not endpoint:
            raise ValueError("HttpRagSystem needs an endpoint URL.")
        self.endpoint = endpoint
        self.timeout = timeout
        self.name = f"http:{endpoint}"

    def ask(self, question: str, k: int = 5) -> SystemResult:
        payload = json.dumps({"question": question, "k": k}).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint, data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"RAG endpoint {self.endpoint} returned HTTP {exc.code} "
                f"({exc.reason}) for question {question!r}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Could not reach RAG endpoint {self.endpoint}: {exc.reason}"
            ) from exc
        try:
            data = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RuntimeError(
                f"RAG endpoint {self.endpoint} returned a body that is not "
                f"valid JSON: {exc}"
            ) from exc
        answer = str(data.get("answer", "") or "")
        abstained = bool(data.get("abstained", data.get("abstain", False)))
        citations: list[Citation] = []
        for raw in data.get("citations", []) or []:
            if isinstance(raw, str):
                citations.append(Citation(species=raw))
            elif isinstance(raw, dict):
                citations.append(
                    Citation(
                        species=raw.get("species") or raw.get("common_name"),
                        source=raw.get("source"),
                        page=raw.get("page"),
                        score=raw.get("score"),
                    )
                )
        retrieved = (
            data.get("retrieved_chunk_ids")
            or data.get("retrieved_ids")
            or data.get("chunk_ids")
            or []
        )
        return SystemResult(
            answer=answer,
            citations=citations,
            abstained=abstained,
            retrieved_chunk_ids=[int(i) for i in retrieved],
        )


class StubSystem:
    """A fake system for tests and smoke runs.

    ``answers`` maps a question to the SystemResult to return; any
    question not in the map gets a fixed fallback answer with the
    default retrieved ids.
    """

    name = "stub"

    def __init__(
        self,
        answers: dict[str, SystemResult] | None = None,
        fallback_retrieved_ids: list[int] | None = None,
    ) -> None:
        self.answers = answers or {}
        self.fallback_retrieved_ids = fallback_retrieved_ids or [0, 1, 2]

    def ask(self, question: str, k: int = 5) -> SystemResult:
        if question in self.answers:
            return self.answers[question]
        return SystemResult(
            answer="I don't know.",
            citations=[],
            abstained=True,
            retrieved_chunk_ids=list(self.fallback_retrieved_ids[:k]),
        )

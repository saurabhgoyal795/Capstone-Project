"""Workflow orchestration: per-document task graph and batch processing.

Per document the workflow is::

    document text
        |
        v
    [1] extract_case  ──(fails)──> FAILED
        |
        v  extraction
    ┌───────────────┬────────────────┐   (RunnableParallel)
    [2a] customer email   [2b] case summary
    └───────────────┴────────────────┘
        |
        v
    SUCCESS (both ok) / PARTIAL (one or both failed)

Documents themselves are processed concurrently with a ThreadPoolExecutor.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableLambda, RunnableParallel

from complaint_processor import chains
from complaint_processor.config import Settings
from complaint_processor.schemas import (
    ComplaintExtraction,
    DocumentResult,
    LoadedDocument,
    ProcessingStatus,
)

logger = logging.getLogger(__name__)


@dataclass
class TaskOutcome:
    """Result of one guarded AI task: either a value or an error message."""

    name: str
    value: Any = None
    error: Optional[str] = None
    duration_seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None


def _run_guarded(name: str, func: Callable[[], Any]) -> TaskOutcome:
    """Run ``func`` and capture its result, timing and any exception."""
    start = time.perf_counter()
    try:
        value = func()
        return TaskOutcome(name=name, value=value, duration_seconds=time.perf_counter() - start)
    except Exception as exc:  # noqa: BLE001 - one task must never crash the batch
        return TaskOutcome(
            name=name,
            error=f"{name} failed: {type(exc).__name__}: {exc}",
            duration_seconds=time.perf_counter() - start,
        )


class CaseProcessingPipeline:
    """Runs the three AI tasks for each complaint document.

    The LLM is injected so tests (or alternative providers) can supply any
    ``BaseChatModel``-compatible object.
    """

    def __init__(self, llm: BaseChatModel, settings: Optional[Settings] = None) -> None:
        self.llm = llm
        self.settings = settings or Settings()
        self._downstream = self._build_downstream_runnable()

    # ------------------------------------------------------------------ graph
    def _build_downstream_runnable(self) -> RunnableParallel:
        """Build the parallel email + summary step as a LangChain RunnableParallel.

        Each branch is wrapped so that a failure in one branch is captured as a
        ``TaskOutcome`` instead of cancelling the sibling branch. Chain functions
        are looked up on the ``chains`` module at call time (monkeypatch-friendly).
        """

        def email_branch(inputs: dict) -> TaskOutcome:
            return _run_guarded(
                "email_generation",
                lambda: chains.generate_customer_email(self.llm, inputs["text"], inputs["extraction"]),
            )

        def summary_branch(inputs: dict) -> TaskOutcome:
            return _run_guarded(
                "case_summary",
                lambda: chains.generate_case_summary(self.llm, inputs["text"], inputs["extraction"]),
            )

        return RunnableParallel(
            email=RunnableLambda(email_branch).with_config(run_name="customer_email"),
            summary=RunnableLambda(summary_branch).with_config(run_name="case_summary"),
        )

    # --------------------------------------------------------------- per doc
    def process_document(self, doc: LoadedDocument) -> DocumentResult:
        """Process one document through extraction and the parallel downstream tasks.

        Never raises: all errors are recorded on the returned ``DocumentResult``.
        """
        start = time.perf_counter()
        result = DocumentResult(doc_id=doc.doc_id, file_name=doc.file_name, status=ProcessingStatus.FAILED)
        logger.info("[%s] starting (%d chars)", doc.doc_id, doc.char_count)

        # Step 1: structured extraction (everything else depends on it).
        extraction_outcome = _run_guarded("extraction", lambda: chains.extract_case(self.llm, doc.text))
        if not extraction_outcome.ok:
            logger.error("[%s] step 1/2 extraction FAILED in %.2fs: %s",
                         doc.doc_id, extraction_outcome.duration_seconds, extraction_outcome.error)
            result.errors.append(extraction_outcome.error or "extraction failed")
            result.duration_seconds = round(time.perf_counter() - start, 3)
            return result

        extraction: ComplaintExtraction = extraction_outcome.value
        result.extraction = extraction
        logger.info("[%s] step 1/2 extraction ok in %.2fs (category=%s, escalation=%s)",
                    doc.doc_id, extraction_outcome.duration_seconds,
                    extraction.complaint_category.value, extraction.escalation_required.value)

        # Step 2: email + summary concurrently.
        step2_start = time.perf_counter()
        try:
            outcomes: dict[str, TaskOutcome] = self._downstream.invoke(
                {"text": doc.text, "extraction": extraction}
            )
        except Exception as exc:  # noqa: BLE001 - defensive: runnable infrastructure error
            outcomes = {
                "email": TaskOutcome("email_generation", error=f"parallel step failed: {exc}"),
                "summary": TaskOutcome("case_summary", error=f"parallel step failed: {exc}"),
            }

        for key, outcome in outcomes.items():
            if outcome.ok:
                setattr(result, key, outcome.value)
                logger.info("[%s] step 2/2 %s ok in %.2fs", doc.doc_id, outcome.name, outcome.duration_seconds)
            else:
                result.errors.append(outcome.error or f"{outcome.name} failed")
                logger.error("[%s] step 2/2 %s FAILED in %.2fs: %s",
                             doc.doc_id, outcome.name, outcome.duration_seconds, outcome.error)
        logger.debug("[%s] step 2/2 wall time %.2fs", doc.doc_id, time.perf_counter() - step2_start)

        result.status = ProcessingStatus.SUCCESS if not result.errors else ProcessingStatus.PARTIAL
        result.duration_seconds = round(time.perf_counter() - start, 3)
        return result

    def _safe_process(self, doc: LoadedDocument) -> DocumentResult:
        """``process_document`` with a last-resort guard so a worker never raises."""
        try:
            return self.process_document(doc)
        except Exception as exc:  # noqa: BLE001
            logger.exception("[%s] unexpected pipeline error", doc.doc_id)
            return DocumentResult(
                doc_id=doc.doc_id,
                file_name=doc.file_name,
                status=ProcessingStatus.FAILED,
                errors=[f"unexpected error: {type(exc).__name__}: {exc}"],
            )

    # ----------------------------------------------------------------- batch
    def run_batch(self, documents: Sequence[LoadedDocument], max_workers: Optional[int] = None) -> list[DocumentResult]:
        """Process many documents concurrently, returning results in input order."""
        docs = list(documents)
        if not docs:
            logger.warning("No documents to process.")
            return []

        workers = max(1, min(max_workers or self.settings.max_workers, len(docs)))
        logger.info("Processing %d document(s) with %d worker(s)", len(docs), workers)

        results: list[Optional[DocumentResult]] = [None] * len(docs)
        completed = 0
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="doc-worker") as pool:
            futures = {pool.submit(self._safe_process, doc): idx for idx, doc in enumerate(docs)}
            for future in as_completed(futures):
                idx = futures[future]
                res = future.result()  # _safe_process never raises
                results[idx] = res
                completed += 1
                logger.info("[%d/%d] %s done (%s, %.1fs)",
                            completed, len(docs), res.file_name, res.status.value, res.duration_seconds)

        return [r for r in results if r is not None]

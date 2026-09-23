"""Command-line entry point: batch-process every complaint document in the data folder.

Usage::

    python main.py                                  # use .env settings
    python main.py --provider ollama --model llama3.1 --workers 4
    python main.py --data-dir data --output-dir output --limit 2 --log-level DEBUG
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

from complaint_processor.config import PROJECT_ROOT, Settings, get_settings
from complaint_processor.ingestion import load_documents
from complaint_processor.llm import get_llm
from complaint_processor.logging_setup import setup_logging
from complaint_processor.pipeline import CaseProcessingPipeline
from complaint_processor.schemas import DocumentResult, ProcessingStatus
from complaint_processor.writers import OutputWriter

logger = logging.getLogger("complaint_processor.cli")

EXIT_OK = 0
EXIT_FATAL = 1
EXIT_NOTHING_PROCESSED = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="complaint-processor",
        description="AI Customer Complaint & Case Processing System - batch-process complaint documents.",
    )
    parser.add_argument("--data-dir", type=Path, help="Folder containing input documents (default: DATA_DIR)")
    parser.add_argument("--output-dir", type=Path, help="Folder for generated outputs (default: OUTPUT_DIR)")
    parser.add_argument("--workers", type=int, help="Documents processed in parallel (default: MAX_WORKERS)")
    parser.add_argument("--provider", choices=["openai", "gemini", "ollama", "bedrock"], help="LLM provider override")
    parser.add_argument("--model", help="Model name override")
    parser.add_argument("--limit", type=int, help="Only process the first N documents (sorted by name)")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], type=str.upper,
                        help="Logging level (default: LOG_LEVEL)")
    return parser


def apply_overrides(settings: Settings, args: argparse.Namespace) -> Settings:
    """Return a copy of ``settings`` with any CLI arguments applied."""
    overrides: dict = {}
    if args.data_dir is not None:
        overrides["data_dir"] = args.data_dir.expanduser().resolve()
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir.expanduser().resolve()
    if args.workers is not None:
        overrides["max_workers"] = args.workers
    if args.provider is not None:
        overrides["llm_provider"] = args.provider
        if args.model is None:
            overrides["model_name"] = ""  # fall back to the provider's default model
    if args.model is not None:
        overrides["model_name"] = args.model
    if args.log_level is not None:
        overrides["log_level"] = args.log_level
    return dataclasses.replace(settings, **overrides) if overrides else settings


def print_summary_table(results: Sequence[DocumentResult], ingestion_errors: Sequence, total_seconds: float,
                        settings: Settings) -> None:
    """Print a compact, human-readable run summary to stdout."""
    rows = [(r.file_name, r.status.value.upper(),
             r.extraction.complaint_category.value if r.extraction else "-",
             r.extraction.escalation_required.value if r.extraction else "-",
             f"{r.duration_seconds:.1f}s",
             (r.errors[0][:60] if r.errors else "")) for r in results]
    rows += [(e.file_name, "FAILED", "-", "-", "0.0s", f"ingestion: {str(e.error)[:49]}") for e in ingestion_errors]
    rows.sort(key=lambda row: row[0])

    headers = ("File", "Status", "Category", "Escalate", "Time", "Error")
    widths = [max(len(h), *(len(str(row[i])) for row in rows)) if rows else len(h) for i, h in enumerate(headers)]
    line = "+-" + "-+-".join("-" * w for w in widths) + "-+"

    def fmt(values: Sequence[str]) -> str:
        return "| " + " | ".join(str(v).ljust(w) for v, w in zip(values, widths)) + " |"

    counts = {s: sum(1 for r in results if r.status == s) for s in ProcessingStatus}
    failed = counts[ProcessingStatus.FAILED] + len(ingestion_errors)

    print()
    print(f"Provider: {settings.llm_provider}  Model: {settings.resolved_model}  Workers: {settings.max_workers}")
    print(line)
    print(fmt(headers))
    print(line)
    for row in rows:
        print(fmt(row))
    print(line)
    print(f"Total: {len(rows)}  Success: {counts[ProcessingStatus.SUCCESS]}  "
          f"Partial: {counts[ProcessingStatus.PARTIAL]}  Failed: {failed}  Time: {total_seconds:.1f}s")
    print(f"Outputs written to: {settings.output_dir}")
    print()


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the full batch workflow. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    settings = apply_overrides(get_settings(), args)

    log_file = setup_logging(settings.log_level, PROJECT_ROOT / "logs")
    logger.info("Starting complaint processing run (log file: %s)", log_file)

    try:
        settings.validate()
    except ValueError as exc:
        logger.error("Configuration error: %s", exc)
        print(f"Configuration error: {exc}", file=sys.stderr)
        return EXIT_FATAL

    if args.limit is not None and args.limit < 1:
        print("--limit must be >= 1", file=sys.stderr)
        return EXIT_FATAL
    if not settings.data_dir.is_dir():
        logger.error("Data directory not found: %s", settings.data_dir)
        print(f"Data directory not found: {settings.data_dir}", file=sys.stderr)
        return EXIT_FATAL

    run_start = time.perf_counter()

    # 1. Ingestion
    documents, ingestion_errors = load_documents(
        settings.data_dir, settings.supported_extensions, settings.max_document_chars
    )
    documents = sorted(documents, key=lambda d: d.file_name)
    if args.limit is not None:
        documents = documents[: args.limit]
    logger.info("Loaded %d document(s); %d ingestion error(s)", len(documents), len(ingestion_errors))
    for err in ingestion_errors:
        logger.warning("Ingestion failed for %s: %s", err.file_name, err.error)

    # 2. LLM
    try:
        llm = get_llm(settings)
    except Exception as exc:  # noqa: BLE001 - surface any provider init failure as fatal config error
        logger.error("Could not initialise LLM (%s/%s): %s", settings.llm_provider, settings.resolved_model, exc)
        print(f"Could not initialise LLM: {exc}", file=sys.stderr)
        return EXIT_FATAL
    logger.info("Using provider=%s model=%s", settings.llm_provider, settings.resolved_model)

    # 3. Pipeline
    writer = OutputWriter(settings.output_dir)
    writer.prepare()
    pipeline = CaseProcessingPipeline(llm, settings)
    results = pipeline.run_batch(documents, max_workers=settings.max_workers)

    # 4. Outputs
    writer.write_results(results)
    total_seconds = time.perf_counter() - run_start
    report_path = writer.write_report(results, ingestion_errors)
    writer.write_run_summary(
        results, ingestion_errors, total_seconds=total_seconds,
        provider=settings.llm_provider, model=settings.resolved_model, max_workers=settings.max_workers,
    )
    logger.info("Run finished in %.1fs; report at %s", total_seconds, report_path)
    print_summary_table(results, ingestion_errors, total_seconds, settings)

    any_ok = any(r.status != ProcessingStatus.FAILED for r in results)
    return EXIT_OK if any_ok else EXIT_NOTHING_PROCESSED


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

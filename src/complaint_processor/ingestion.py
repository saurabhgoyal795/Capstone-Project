"""Document ingestion: discover files in the data folder and extract their text.

Supported formats are plain text (.txt), PDF (.pdf, via pypdf) and Word (.docx,
via python-docx). Any failure on an individual file is captured as an
``IngestionError`` so a single bad file never stops the batch.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from pydantic import BaseModel

from complaint_processor.schemas import LoadedDocument

logger = logging.getLogger(__name__)

__all__ = ["IngestionError", "extract_text", "load_documents"]


class IngestionError(BaseModel):
    """A file that could not be read or yielded no usable text."""

    file_name: str
    path: str
    error: str


# --------------------------------------------------------------------------- #
# Per-format extractors
# --------------------------------------------------------------------------- #
def _extract_txt(path: Path) -> str:
    """Read a text file as UTF-8, falling back to latin-1 for legacy encodings."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        logger.debug("UTF-8 decode failed for %s; retrying with latin-1", path.name)
        return path.read_text(encoding="latin-1")


def _extract_pdf(path: Path) -> str:
    """Concatenate the extracted text of every page in a PDF."""
    from pypdf import PdfReader  # local import keeps module import cheap

    reader = PdfReader(str(path))
    if reader.is_encrypted:
        # Attempt an empty-password decrypt; many "encrypted" PDFs allow this.
        try:
            reader.decrypt("")
        except Exception as exc:  # noqa: BLE001 - surface as a clear message
            raise ValueError(f"PDF is encrypted and cannot be decrypted: {exc}") from exc
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def _extract_docx(path: Path) -> str:
    """Extract paragraph and table-cell text from a .docx file, in document order.

    Walking the body in order keeps each table next to the heading it belongs to,
    which gives the LLM much better context than appending all tables at the end.
    """
    import docx  # python-docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = docx.Document(str(path))
    parts: list[str] = []
    for block in document.element.body.iterchildren():
        tag = block.tag.rsplit("}", 1)[-1]
        if tag == "p":
            text = Paragraph(block, document).text
            if text.strip():
                parts.append(text)
        elif tag == "tbl":
            for row in Table(block, document).rows:
                # De-duplicate merged cells, which python-docx repeats per grid column.
                cells: list[str] = []
                for cell in row.cells:
                    text = cell.text.strip()
                    if text and (not cells or cells[-1] != text):
                        cells.append(text)
                if cells:
                    parts.append(" | ".join(cells))
    return "\n".join(parts)


_EXTRACTORS = {
    ".txt": _extract_txt,
    ".pdf": _extract_pdf,
    ".docx": _extract_docx,
}


def extract_text(path: Path) -> str:
    """Extract raw text from ``path`` based on its file extension.

    Raises:
        ValueError: if the extension is unsupported or no text could be extracted.
    """
    path = Path(path)
    ext = path.suffix.lower()
    extractor = _EXTRACTORS.get(ext)
    if extractor is None:
        raise ValueError(f"Unsupported file extension '{ext}' for {path.name}")
    text = extractor(path)
    if not text or not text.strip():
        raise ValueError(f"No extractable text found in {path.name}")
    return text


# --------------------------------------------------------------------------- #
# Batch loading
# --------------------------------------------------------------------------- #
_INLINE_WS = re.compile(r"[ \t\f\v ]+")
_MULTI_BLANK = re.compile(r"\n{3,}")


def _normalise_whitespace(text: str) -> str:
    """Collapse runs of spaces/tabs, trim each line and limit blank lines to one."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [_INLINE_WS.sub(" ", line).strip() for line in text.split("\n")]
    return _MULTI_BLANK.sub("\n\n", "\n".join(lines)).strip()


def load_documents(
    data_dir: Path,
    supported_extensions: tuple[str, ...],
    max_chars: int,
) -> tuple[list[LoadedDocument], list[IngestionError]]:
    """Load every supported document in ``data_dir`` (non-recursive, sorted by name).

    Hidden files and unsupported extensions are skipped. Files that fail to parse
    are returned as ``IngestionError`` entries instead of raising.

    Args:
        data_dir: Folder containing input documents.
        supported_extensions: Extensions (with leading dot) to process, e.g. (".pdf",).
        max_chars: Maximum characters kept per document; longer text is truncated.

    Returns:
        A tuple of (loaded documents, per-file errors).

    Raises:
        FileNotFoundError: if ``data_dir`` does not exist or is not a directory.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    allowed = {ext.lower() for ext in supported_extensions}
    documents: list[LoadedDocument] = []
    errors: list[IngestionError] = []

    for path in sorted(data_dir.iterdir(), key=lambda p: p.name.lower()):
        if not path.is_file() or path.name.startswith("."):
            continue
        ext = path.suffix.lower()
        if ext not in allowed:
            logger.info("Skipping %s: unsupported extension '%s'", path.name, ext or "<none>")
            continue

        try:
            text = _normalise_whitespace(extract_text(path))
            if not text:
                raise ValueError(f"No extractable text found in {path.name}")
        except Exception as exc:  # noqa: BLE001 - file-level errors must not crash the batch
            message = f"{type(exc).__name__}: {exc}"
            logger.error("Failed to ingest %s: %s", path.name, message)
            errors.append(IngestionError(file_name=path.name, path=str(path), error=message))
            continue

        if max_chars > 0 and len(text) > max_chars:
            logger.warning(
                "Truncating %s from %d to %d characters", path.name, len(text), max_chars
            )
            text = text[:max_chars]

        documents.append(
            LoadedDocument(
                doc_id=path.stem,
                file_name=path.name,
                file_type=ext.lstrip("."),
                path=path,
                text=text,
                char_count=len(text),
            )
        )
        logger.info("Loaded %s (%d chars)", path.name, len(text))

    logger.info("Ingestion complete: %d loaded, %d failed", len(documents), len(errors))
    return documents, errors

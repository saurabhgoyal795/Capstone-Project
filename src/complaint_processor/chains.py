"""LangChain chains for the three AI tasks, all returning validated Pydantic models.

Workflow:
    document ──► extract_case ──► ComplaintExtraction
                                    │
          ┌─────────────────────────┴─────────────────────────┐
          ▼                                                   ▼
    generate_customer_email ──► CustomerEmail     generate_case_summary ──► CaseSummary

Structured output strategy (see ``structured_call``):
    1. Native: ``llm.with_structured_output(Model)`` (OpenAI/Gemini tool/JSON-schema mode,
       Ollama grammar-constrained ``format=<json schema>``).
    2. Fallback: prompt with ``PydanticOutputParser`` format instructions, repair the raw text
       (strip <think> blocks / code fences, take the first {...} object), validate with Pydantic,
       and retry once feeding the validation error back to the model.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel
from pydantic import BaseModel, ValidationError

from .prompts import EMAIL_PROMPT, EXTRACTION_PROMPT, SUMMARY_PROMPT
from .schemas import CaseSummary, ComplaintExtraction, CustomerEmail

logger = logging.getLogger(__name__)

M = TypeVar("M", bound=BaseModel)


class StructuredOutputError(RuntimeError):
    """Raised when the model output cannot be turned into the requested Pydantic model."""


# --------------------------------------------------------------------------- JSON repair

_THINK_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


def _message_text(message: Any) -> str:
    """Plain text from an AIMessage / str (handles list-of-parts content)."""
    content = getattr(message, "content", message)
    if isinstance(content, list):
        content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


def _first_json_object(text: str) -> str | None:
    """Return the first balanced {...} block in ``text`` (brace matching aware of strings)."""
    start = text.find("{")
    while start != -1:
        depth, in_str, escape = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        start = text.find("{", start + 1)
    return None


def repair_json_text(raw: str) -> str:
    """Best-effort cleanup of LLM output into a JSON object string.

    Removes <think>...</think> reasoning (and an unterminated leading one), markdown code
    fences, and surrounding prose; then extracts the first balanced JSON object.
    """
    text = _THINK_RE.sub("", raw)
    if "</think>" in text:  # opening tag was swallowed by the template; drop everything before close
        text = text.split("</think>", 1)[1]
    fence = _FENCE_RE.search(text)
    if fence:
        text = fence.group(1)
    obj = _first_json_object(text)
    if obj is None:
        raise StructuredOutputError(f"No JSON object found in model output: {raw[:200]!r}")
    # Tolerate trailing commas, a common small-model mistake.
    return re.sub(r",\s*([}\]])", r"\1", obj)


def parse_model_output(raw: str, model_cls: type[M]) -> M:
    """Repair + validate raw LLM text into ``model_cls`` (raises ValidationError/StructuredOutputError)."""
    cleaned = repair_json_text(raw)
    data = json.loads(cleaned)
    # Some models wrap the object: {"properties": {...}} or {"ComplaintExtraction": {...}}.
    if isinstance(data, dict) and len(data) == 1:
        (only_value,) = data.values()
        if isinstance(only_value, dict) and set(only_value) & set(model_cls.model_fields):
            data = only_value
    return model_cls.model_validate(data)


# --------------------------------------------------------------------------- structured call


def _native_method(llm: BaseChatModel) -> str | None:
    """Pick the with_structured_output method best suited to the provider."""
    name = type(llm).__name__
    if name == "ChatOllama":
        return "json_schema"  # grammar-constrained decoding via Ollama `format=<schema>`
    return None  # provider default (OpenAI: json_schema/tools, Gemini: tool calling)


def _native_call(llm: BaseChatModel, prompt: ChatPromptTemplate, model_cls: type[M], inputs: dict) -> M:
    method = _native_method(llm)
    kwargs = {"method": method} if method else {}
    structured = llm.with_structured_output(model_cls, **kwargs)
    result = (prompt | structured).invoke(inputs)
    if isinstance(result, dict):  # some integrations return dicts
        result = model_cls.model_validate(result)
    if not isinstance(result, model_cls):
        raise StructuredOutputError(f"Native structured output returned {type(result).__name__}")
    return result


def _fallback_call(llm: BaseChatModel, prompt: ChatPromptTemplate, model_cls: type[M], inputs: dict) -> M:
    parser = PydanticOutputParser(pydantic_object=model_cls)
    instructions = (
        parser.get_format_instructions()
        + "\nReturn ONLY the JSON object. No markdown, no explanations, no <think> text."
    )
    messages: list[BaseMessage] = prompt.format_messages(**inputs, format_instructions=instructions)

    raw = _message_text(llm.invoke(messages))
    try:
        return parse_model_output(raw, model_cls)
    except (ValidationError, StructuredOutputError, json.JSONDecodeError) as first_error:
        logger.warning("Fallback parse failed for %s, retrying with error feedback: %s",
                       model_cls.__name__, _short(first_error))
        retry_messages = messages + [
            AIMessage(content=raw),
            HumanMessage(
                content=(
                    "Your previous reply could not be parsed/validated:\n"
                    f"{_short(first_error, 1500)}\n\n"
                    "Reply again with ONLY a single corrected JSON object that matches the schema. "
                    "Use exactly the allowed enum values."
                )
            ),
        ]
        raw = _message_text(llm.invoke(retry_messages))
        try:
            return parse_model_output(raw, model_cls)
        except (ValidationError, StructuredOutputError, json.JSONDecodeError) as second_error:
            raise StructuredOutputError(
                f"Could not produce a valid {model_cls.__name__} after retry: {_short(second_error)}"
            ) from second_error


def _short(err: Exception, limit: int = 300) -> str:
    text = str(err).replace("\n", " ") if limit <= 300 else str(err)
    return text[:limit]


def structured_call(llm: BaseChatModel, prompt: ChatPromptTemplate, model_cls: type[M], inputs: dict) -> M:
    """Run ``prompt`` on ``llm`` and return a validated ``model_cls`` instance.

    Tries native structured output first; on any failure (unsupported, refusal, invalid JSON,
    schema mismatch) falls back to format-instructions + JSON repair + one validation retry.
    """
    try:
        return _native_call(llm, prompt, model_cls, inputs)
    except NotImplementedError:
        logger.debug("%s has no native structured output; using fallback", type(llm).__name__)
    except Exception as exc:  # noqa: BLE001 - any native failure should trigger the fallback
        logger.warning("Native structured output failed for %s (%s); using fallback parser",
                       model_cls.__name__, _short(exc))
    return _fallback_call(llm, prompt, model_cls, inputs)


# --------------------------------------------------------------------------- guardrails

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"(?<!\w)\+?\d[\d\s().-]{6,}\d(?!\w)")
_PLACEHOLDER_RE = re.compile(r"\[(?:[A-Z][\w ]{1,30})\]|<(?:date|name|number|company)[^>]*>", re.IGNORECASE)


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


def check_email_guardrails(email: CustomerEmail, document_text: str) -> list[str]:
    """Return a list of guardrail violations found in the generated email (empty = OK).

    Flags email addresses / phone numbers not present in the source document (likely invented)
    and unfilled template placeholders such as "[Customer Name]".
    """
    issues: list[str] = []
    text = f"{email.subject}\n{email.body}"
    doc_lower = document_text.lower()
    doc_digits = _digits(document_text)

    for addr in sorted(set(_EMAIL_RE.findall(text))):
        if addr.lower() not in doc_lower:
            issues.append(f"email address not in source document: {addr}")
    for phone in sorted(set(_PHONE_RE.findall(text))):
        digits = _digits(phone)
        if len(digits) >= 7 and digits not in doc_digits:
            issues.append(f"phone number not in source document: {phone.strip()}")
    for ph in sorted(set(_PLACEHOLDER_RE.findall(text))):
        issues.append(f"unfilled placeholder: {ph}")
    return issues


def _sanitise_extraction(extraction: ComplaintExtraction, document_text: str) -> ComplaintExtraction:
    """Null out contact details the model returned that do not literally appear in the document."""
    updates: dict[str, Any] = {}
    if extraction.email and extraction.email.lower() not in document_text.lower():
        logger.warning("Extracted email %r not found in document; setting to null", extraction.email)
        updates["email"] = None
    if extraction.phone_number:
        digits = _digits(extraction.phone_number)
        if not digits or digits not in _digits(document_text):
            logger.warning("Extracted phone %r not found in document; setting to null", extraction.phone_number)
            updates["phone_number"] = None
    return extraction.model_copy(update=updates) if updates else extraction


# --------------------------------------------------------------------------- chains


def build_extraction_chain(llm: BaseChatModel) -> Runnable:
    """Runnable: {"document_text": str} -> ComplaintExtraction."""

    def _run(inputs: dict) -> ComplaintExtraction:
        result = structured_call(llm, EXTRACTION_PROMPT, ComplaintExtraction, inputs)
        return _sanitise_extraction(result, inputs["document_text"])

    return RunnableLambda(_run, name="extract_case")


def build_email_chain(llm: BaseChatModel) -> Runnable:
    """Runnable: {"document_text": str, "extraction_json": str} -> CustomerEmail (guardrail-checked)."""

    def _run(inputs: dict) -> CustomerEmail:
        email = structured_call(llm, EMAIL_PROMPT, CustomerEmail, inputs)
        issues = check_email_guardrails(email, inputs["document_text"])
        for issue in issues:
            logger.warning("Email guardrail: %s", issue)
        return email

    return RunnableLambda(_run, name="generate_customer_email")


def build_summary_chain(llm: BaseChatModel) -> Runnable:
    """Runnable: {"document_text": str, "extraction_json": str} -> CaseSummary."""

    def _run(inputs: dict) -> CaseSummary:
        return structured_call(llm, SUMMARY_PROMPT, CaseSummary, inputs)

    return RunnableLambda(_run, name="generate_case_summary")


def build_post_extraction_parallel(llm: BaseChatModel) -> RunnableParallel:
    """Run email + summary generation concurrently after extraction.

    Input dict:  {"document_text": str, "extraction_json": str}
                 (use ``extraction_to_json(extraction)`` for the second key)
    Output dict: {"email": CustomerEmail, "summary": CaseSummary}

    Note: if either branch raises, ``invoke`` raises; callers wanting partial results should
    call ``build_email_chain`` / ``build_summary_chain`` separately.
    """
    return RunnableParallel(email=build_email_chain(llm), summary=build_summary_chain(llm))


# --------------------------------------------------------------------------- public API


def extraction_to_json(extraction: ComplaintExtraction) -> str:
    """Serialise an extraction for the downstream prompts."""
    return extraction.model_dump_json(indent=2)


def extract_case(llm: BaseChatModel, document_text: str) -> ComplaintExtraction:
    """Task 1: document -> structured ComplaintExtraction."""
    return build_extraction_chain(llm).invoke({"document_text": document_text})


def generate_customer_email(llm: BaseChatModel, document_text: str, extraction: ComplaintExtraction) -> CustomerEmail:
    """Task 2: document + extraction -> CustomerEmail."""
    return build_email_chain(llm).invoke(
        {"document_text": document_text, "extraction_json": extraction_to_json(extraction)}
    )


def generate_case_summary(llm: BaseChatModel, document_text: str, extraction: ComplaintExtraction) -> CaseSummary:
    """Task 3: document + extraction -> internal CaseSummary."""
    return build_summary_chain(llm).invoke(
        {"document_text": document_text, "extraction_json": extraction_to_json(extraction)}
    )

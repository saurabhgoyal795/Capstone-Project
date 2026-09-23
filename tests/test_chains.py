"""Offline tests for chains.py: fallback parsing, JSON repair, guardrails. No network."""

from __future__ import annotations

import json
import logging

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from complaint_processor.chains import (
    StructuredOutputError,
    build_post_extraction_parallel,
    check_email_guardrails,
    extract_case,
    extraction_to_json,
    generate_case_summary,
    generate_customer_email,
    parse_model_output,
    repair_json_text,
)
from complaint_processor.schemas import (
    CaseStatus,
    CaseSummary,
    ComplaintCategory,
    ComplaintExtraction,
    CustomerEmail,
    YesNo,
)

DOC = (
    "From: Jane Doe <jane.doe@example.com>\n"
    "My Acme router keeps dropping the connection every hour. Screenshot attached. "
    "Call me on 555-123-4567. I want a manager if this is not fixed.\n"
)

EXTRACTION = {
    "customer_name": "Jane Doe",
    "email": "jane.doe@example.com",
    "phone_number": "555-123-4567",
    "product_or_service": "Acme router",
    "complaint_category": "Technical Issue",
    "issue_description": "Router drops the connection every hour.",
    "resolution_provided": None,
    "is_complaint": "Yes",
    "escalation_required": "Yes",
    "supporting_document_available": "Yes",
    "overall_case_status": "Open",
}
EMAIL = {"subject": "Your router connection issue", "body": "Dear Jane Doe,\n...\nKind regards,\nCustomer Care Team, Acme"}
SUMMARY = {
    "case_overview": "Jane Doe reports an Acme router issue.",
    "key_issue": "Hourly connection drops.",
    "action_taken": "No action recorded in the document.",
    "current_status": "Open",
    "recommended_next_action": "Technical team to diagnose the router.",
}


def fake(*responses: str) -> FakeListChatModel:
    # FakeListChatModel has no native structured output -> exercises the fallback path.
    return FakeListChatModel(responses=list(responses))


# --------------------------------------------------------------------------- JSON repair


def test_repair_strips_think_and_code_fence():
    raw = '<think>Let me reason {not json}</think>\nHere you go:\n```json\n{"a": 1, "b": "x}"}\n```\nDone.'
    assert json.loads(repair_json_text(raw)) == {"a": 1, "b": "x}"}


def test_repair_handles_orphan_think_close_and_trailing_commas():
    raw = 'thinking without opening tag {x}</think> {"a": [1, 2,], "b": {"c": 3,},}'
    assert json.loads(repair_json_text(raw)) == {"a": [1, 2], "b": {"c": 3}}


def test_repair_raises_when_no_json():
    with pytest.raises(StructuredOutputError):
        repair_json_text("I cannot help with that.")


def test_parse_model_output_unwraps_single_key_wrapper():
    raw = json.dumps({"CaseSummary": SUMMARY})
    assert parse_model_output(raw, CaseSummary).key_issue == "Hourly connection drops."


# --------------------------------------------------------------------------- fallback path


def test_extract_case_fallback_parses_noisy_output():
    llm = fake(f"<think>hmm</think>\n```json\n{json.dumps(EXTRACTION)}\n```")
    result = extract_case(llm, DOC)
    assert isinstance(result, ComplaintExtraction)
    assert result.complaint_category is ComplaintCategory.TECHNICAL_ISSUE
    assert result.escalation_required is YesNo.YES
    assert result.overall_case_status is CaseStatus.OPEN


def test_fallback_retries_once_after_validation_error():
    bad = dict(EXTRACTION, complaint_category="Connectivity")  # not an allowed enum value
    llm = fake(json.dumps(bad), json.dumps(EXTRACTION))
    result = extract_case(llm, DOC)
    assert result.complaint_category is ComplaintCategory.TECHNICAL_ISSUE
    assert llm.i == 0  # both scripted responses consumed (index wrapped around)


def test_fallback_raises_after_second_failure():
    llm = fake("not json at all", '{"subject": "only subject"}')
    with pytest.raises(StructuredOutputError):
        generate_customer_email(llm, DOC, ComplaintExtraction.model_validate(EXTRACTION))


def test_extraction_nulls_contact_details_not_in_document():
    hallucinated = dict(EXTRACTION, email="someone@else.com", phone_number="+1 999 888 7777")
    result = extract_case(fake(json.dumps(hallucinated)), DOC)
    assert result.email is None
    assert result.phone_number is None
    assert result.customer_name == "Jane Doe"


def test_email_and_summary_functions_return_models():
    extraction = ComplaintExtraction.model_validate(EXTRACTION)
    email = generate_customer_email(fake(json.dumps(EMAIL)), DOC, extraction)
    summary = generate_case_summary(fake(json.dumps(SUMMARY)), DOC, extraction)
    assert isinstance(email, CustomerEmail) and email.body.startswith("Dear Jane Doe")
    assert isinstance(summary, CaseSummary)


class RoutingFakeChat(FakeListChatModel):
    """Answers by task (detected from the system prompt) so concurrent branches are deterministic."""

    def _call(self, messages, stop=None, run_manager=None, **kwargs):
        system = messages[0].content
        return json.dumps(EMAIL if "reply email" in system else SUMMARY)


def test_post_extraction_parallel_returns_both_models():
    chain = build_post_extraction_parallel(RoutingFakeChat(responses=["unused"]))
    out = chain.invoke(
        {"document_text": DOC, "extraction_json": extraction_to_json(ComplaintExtraction.model_validate(EXTRACTION))}
    )
    assert isinstance(out["email"], CustomerEmail) and out["email"].subject == EMAIL["subject"]
    assert isinstance(out["summary"], CaseSummary) and out["summary"].key_issue == SUMMARY["key_issue"]


# --------------------------------------------------------------------------- guardrails


def test_guardrail_passes_for_contacts_present_in_document():
    email = CustomerEmail(subject="Re: router", body="We will contact you at jane.doe@example.com or 555 123 4567.")
    assert check_email_guardrails(email, DOC) == []


def test_guardrail_flags_invented_contacts_and_placeholders():
    email = CustomerEmail(
        subject="Re: router",
        body="Dear [Customer Name], call 1-800-555-0199 or write to help@acme-support.com.",
    )
    issues = check_email_guardrails(email, DOC)
    assert any("help@acme-support.com" in i for i in issues)
    assert any("1-800-555-0199" in i for i in issues)
    assert any("placeholder" in i for i in issues)


def test_email_chain_logs_guardrail_warning_but_returns_email(caplog):
    bad = {"subject": "Re: router", "body": "Please call 1-800-555-0199."}
    with caplog.at_level(logging.WARNING, logger="complaint_processor.chains"):
        email = generate_customer_email(fake(json.dumps(bad)), DOC, ComplaintExtraction.model_validate(EXTRACTION))
    assert email.body == bad["body"]
    assert "Email guardrail" in caplog.text

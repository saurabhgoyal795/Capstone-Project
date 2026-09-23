"""Pydantic schemas: the contract between ingestion, LLM chains, pipeline and writers."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class YesNo(str, Enum):
    YES = "Yes"
    NO = "No"


class ComplaintCategory(str, Enum):
    BILLING = "Billing"
    PRODUCT_DEFECT = "Product Defect"
    SERVICE_QUALITY = "Service Quality"
    DELIVERY = "Delivery"
    TECHNICAL_ISSUE = "Technical Issue"
    REFUND = "Refund"
    ACCOUNT = "Account"
    OTHER = "Other"


class CaseStatus(str, Enum):
    OPEN = "Open"
    IN_PROGRESS = "In Progress"
    RESOLVED = "Resolved"
    ESCALATED = "Escalated"
    CLOSED = "Closed"


class LoadedDocument(BaseModel):
    """A document successfully read from the data folder."""

    doc_id: str = Field(description="File stem, e.g. complaint_001")
    file_name: str
    file_type: str = Field(description="Extension without dot: txt | pdf | docx")
    path: Path
    text: str
    char_count: int


class ComplaintExtraction(BaseModel):
    """Structured fields the LLM must extract from one complaint document."""

    customer_name: Optional[str] = Field(None, description="Full name of the customer, null if absent")
    email: Optional[str] = Field(None, description="Customer email address, null if absent")
    phone_number: Optional[str] = Field(None, description="Customer phone number, null if absent")
    product_or_service: Optional[str] = Field(None, description="Product or service involved")
    complaint_category: ComplaintCategory = Field(description="Best-fit category for the issue")
    issue_description: str = Field(description="1-3 sentence factual description of the issue")
    resolution_provided: Optional[str] = Field(
        None, description="Resolution already provided per the document, null if none"
    )
    is_complaint: YesNo = Field(description="Yes if the document is a customer complaint")
    escalation_required: YesNo = Field(description="Yes if the document indicates escalation is needed/requested")
    supporting_document_available: YesNo = Field(
        description="Yes if the document mentions attachments/receipts/screenshots/evidence"
    )
    overall_case_status: CaseStatus = Field(description="Current status of the case per the document")

    @field_validator("email")
    @classmethod
    def _normalise_email(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip().lower()
        return v if "@" in v else None


class CustomerEmail(BaseModel):
    subject: str
    body: str = Field(description="Full email body, greeting to sign-off, plain text")


class CaseSummary(BaseModel):
    case_overview: str
    key_issue: str
    action_taken: str
    current_status: str
    recommended_next_action: str


class ProcessingStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"  # extraction ok, a downstream task failed
    FAILED = "failed"


class DocumentResult(BaseModel):
    """Everything produced for one document; the pipeline returns one per input file."""

    doc_id: str
    file_name: str
    status: ProcessingStatus
    extraction: Optional[ComplaintExtraction] = None
    email: Optional[CustomerEmail] = None
    summary: Optional[CaseSummary] = None
    errors: list[str] = Field(default_factory=list)
    duration_seconds: float = 0.0

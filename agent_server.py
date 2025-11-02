"""FastAPI backend exposing SBA Copilot agent capabilities.

This lightweight service wraps the retrieval + synthesis primitives in
``agent_tools`` so downstream clients (Streamlit UI, CRM integrations, etc.)
can reuse the same logic.
"""

from __future__ import annotations

import pathlib
from typing import Any, Dict, Iterable, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent_tools import (
    MissingArtifactError,
    answer_query,
    draft_email,
    log_crm_note,
    open_source,
    rerank_hits,
    search_docs,
)
from underwriter.tools import (
    calc_cashflow,
    compute_decision,
    generate_letter,
    policy_check as uw_policy_check,
    risk_score as uw_risk_score,
    underwrite_case,
)

app = FastAPI(title="SBA Copilot Agent", version="0.1.0")


class QueryRequest(BaseModel):
    question: str = Field(..., description="Natural language SBA question")
    k: int = Field(40, ge=1, le=200)
    topn: int = Field(6, ge=1, le=20)


class SourceResponse(BaseModel):
    text: str
    metadata: Dict[str, Any]
    citation: str


class AnswerResponse(BaseModel):
    question: str
    answer: str
    latency_ms: int
    confidence: Optional[float] = None
    sources: List[SourceResponse]


class CashflowRequest(BaseModel):
    transactions_csv: str = Field(..., description="Path to CSV with columns date, amount")
    loan_terms: Dict[str, Any] = Field(
        default_factory=dict,
        description="Loan amount/rate/term for DSCR calculations",
    )


class CashflowResponse(BaseModel):
    monthly_cashflow: Dict[str, Dict[str, float]]
    ttm_revenue: float
    ebitda: float
    dscr: float
    annual_debt_service: float


class PolicyCheckRequest(BaseModel):
    program: str
    facts: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)


class PolicyCheckResponse(BaseModel):
    passes: bool
    failed_rules: List[Dict[str, Any]]
    decision: str
    conditions: List[str]


class RiskScoreRequest(BaseModel):
    features: Dict[str, Any]


class RiskScoreResponse(BaseModel):
    pd: float
    lgd: float
    risk_band: str
    contributions: Dict[str, float]
    collateral_coverage: float | None = None
    dscr: float | None = None


class UnderwriteRequest(BaseModel):
    program: str
    features: Dict[str, Any]
    loan_terms: Dict[str, Any] = Field(default_factory=dict)
    transactions_csv: Optional[str] = Field(
        default=None,
        description="Optional CSV path of bank transactions for DSCR",
    )
    citation_query: Optional[str] = None
    recipient: Optional[str] = None
    finalize: bool = False


class UnderwriteResponse(BaseModel):
    program: str
    decision: str
    policy: Dict[str, Any]
    risk: Dict[str, Any]
    metrics: Dict[str, Any]
    conditions: List[str]
    quality: float
    citations: List[Dict[str, Any]]
    letter: Dict[str, Any]
    cashflow: Optional[Dict[str, Any]] = None
    audit: Dict[str, str]


class SearchResponse(BaseModel):
    hits: List[SourceResponse]


class DraftEmailRequest(BaseModel):
    recipient: str
    subject: str
    bullet_points: Iterable[str]


class DraftEmailResponse(BaseModel):
    message: str


class LogNoteRequest(BaseModel):
    payload: Dict[str, Any]
    path: Optional[str] = Field(
        None,
        description="Optional override path for CRM note log (JSONL file)",
    )


class LogNoteResponse(BaseModel):
    path: str


@app.get("/health")
def healthcheck() -> Dict[str, str]:
    """Simple liveness probe."""
    return {"status": "ok"}


@app.post("/answer", response_model=AnswerResponse)
def answer_endpoint(req: QueryRequest) -> AnswerResponse:
    try:
        result = answer_query(req.question, k=req.k, topn=req.topn)
    except MissingArtifactError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    sources = [SourceResponse(**src) for src in result.get("sources", [])]
    return AnswerResponse(
        question=result["question"],
        answer=result["answer"],
        latency_ms=result["latency_ms"],
        confidence=result.get("quality"),
        sources=sources,
    )


@app.post("/search", response_model=SearchResponse)
def search_endpoint(req: QueryRequest) -> SearchResponse:
    try:
        hits = search_docs(req.question, k=req.k)
    except MissingArtifactError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    sources = [SourceResponse(**open_source(hit)) for hit in hits[: req.topn]]
    return SearchResponse(hits=sources)


@app.post("/calc_cashflow", response_model=CashflowResponse)
def calc_cashflow_endpoint(req: CashflowRequest) -> CashflowResponse:
    result = calc_cashflow(req.transactions_csv, req.loan_terms)
    return CashflowResponse(**result)


@app.post("/policy_check", response_model=PolicyCheckResponse)
def policy_check_endpoint(req: PolicyCheckRequest) -> PolicyCheckResponse:
    try:
        result = uw_policy_check(req.program, req.facts, req.metrics)
    except MissingArtifactError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return PolicyCheckResponse(**result)


@app.post("/risk_score", response_model=RiskScoreResponse)
def risk_score_endpoint(req: RiskScoreRequest) -> RiskScoreResponse:
    result = uw_risk_score(req.features)
    return RiskScoreResponse(**result)


@app.post("/underwrite", response_model=UnderwriteResponse)
def underwrite_endpoint(req: UnderwriteRequest) -> UnderwriteResponse:
    try:
        result = underwrite_case(
            program=req.program,
            features=req.features,
            loan_terms=req.loan_terms,
            transactions_csv=req.transactions_csv,
            citation_query=req.citation_query,
            recipient=req.recipient,
            finalize=req.finalize,
        )
    except MissingArtifactError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return UnderwriteResponse(**result)


@app.post("/draft_email", response_model=DraftEmailResponse)
def draft_email_endpoint(req: DraftEmailRequest) -> DraftEmailResponse:
    message = draft_email(req.recipient, req.subject, req.bullet_points)
    return DraftEmailResponse(message=message)


@app.post("/log_note", response_model=LogNoteResponse)
def log_note_endpoint(req: LogNoteRequest) -> LogNoteResponse:
    path = pathlib.Path(req.path) if req.path else None
    written_path = log_crm_note(req.payload, path)
    return LogNoteResponse(path=str(written_path))


@app.on_event("startup")
def warm_cache() -> None:
    """Optionally warm RAG artifacts on startup for lower latency."""
    try:
        search_docs("SBA startup probe", k=1)
    except MissingArtifactError:
        # Ignore warming errors; endpoint will raise when accessed.
        pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("agent_server:app", host="0.0.0.0", port=8000, reload=False)

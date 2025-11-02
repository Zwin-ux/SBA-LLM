"""Underwriter Copilot tool implementations.

These functions provide deterministic analytics, rule checks, and document
extraction that the agent layer can call. They are designed to work alongside
`agent_tools` retrieval primitives.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import yaml

from agent_tools import MissingArtifactError, quality_score, search_docs
from underwriter.pd_model import PDModelConfig, load_config, probability_of_default

DATA_DIR = Path("rag/data")
RULES_PATH = Path("config/rules.yaml")
AUDIT_DIR = Path("logs/audit")


@dataclass
class AuditEntry:
    tool: str
    input_hash: str
    output_hash: str
    payload: Dict[str, Any]


def _hash_payload(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def log_audit(tool: str, payload: Dict[str, Any]) -> AuditEntry:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    hashed_input = _hash_payload(payload.get("input"))
    hashed_output = _hash_payload(payload.get("output"))
    entry = {
        "tool": tool,
        "input_hash": hashed_input,
        "output_hash": hashed_output,
        "timestamp": payload.get("timestamp"),
        "metadata": payload.get("metadata", {}),
    }
    target = AUDIT_DIR / f"{tool}.jsonl"
    with open(target, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return AuditEntry(tool=tool, input_hash=hashed_input, output_hash=hashed_output, payload=entry)


def cite_sources(claim_text: str, query: Optional[str] = None, topn: int = 3) -> List[Dict[str, Any]]:
    query = query or claim_text
    hits = search_docs(query, k=40)
    top_hits = hits[:topn]
    return [
        {
            "doc": hit.metadata.get("doc"),
            "title": hit.metadata.get("title"),
            "page": hit.metadata.get("page"),
            "score": hit.score,
            "text": hit.text,
        }
        for hit in top_hits
    ]


def extract_docs(files: Sequence[str], doc_types: Sequence[str]) -> Dict[str, Any]:
    # Placeholder: in production you'd swap in OCR/PDF parsers.
    extracted: List[Dict[str, Any]] = []
    for file_path, doc_type in zip(files, doc_types):
        extracted.append(
            {
                "file": file_path,
                "doc_type": doc_type,
                "entities": [
                    {
                        "type": "placeholder",
                        "value": "N/A",
                        "page": 1,
                        "confidence": 0.0,
                    }
                ],
            }
        )
    return {"documents": extracted}


def calc_cashflow(transactions_csv: str, loan_terms: Dict[str, Any]) -> Dict[str, Any]:
    df = pd.read_csv(transactions_csv)
    if not {"date", "amount"}.issubset(df.columns):
        raise ValueError("CSV must include 'date' and 'amount' columns")
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M")
    monthly = df.groupby("month")["amount"].sum().sort_index()

    inflow = monthly.clip(lower=0)
    outflow = (-monthly.clip(upper=0)).abs()
    monthly_cashflow = {
        str(month): {"in": float(inflow.get(month, 0.0)), "out": float(outflow.get(month, 0.0))}
        for month in monthly.index
    }

    ttm_revenue = float(inflow.sum())
    ebitda = float((inflow - outflow * 0.4).sum())  # simple margin assumption

    rate = float(loan_terms.get("rate", 0.1)) / 12.0
    term = int(loan_terms.get("term_months", 120))
    amount = float(loan_terms.get("amount", 0.0))
    if rate == 0:
        monthly_payment = amount / term if term else 0.0
    else:
        monthly_payment = np.pmt(rate, term, -amount)
    annual_debt_service = float(monthly_payment * 12)
    dscr = round(ebitda / annual_debt_service, 3) if annual_debt_service else float("inf")

    result = {
        "monthly_cashflow": monthly_cashflow,
        "ttm_revenue": round(ttm_revenue, 2),
        "ebitda": round(ebitda, 2),
        "dscr": dscr,
        "annual_debt_service": round(annual_debt_service, 2),
    }
    return result


def collateral_coverage(assets: Dict[str, float], loan_amount: float) -> float:
    total = (
        0.9 * float(assets.get("real_estate", 0.0))
        + 0.8 * float(assets.get("equipment", 0.0))
        + 0.5 * float(assets.get("inventory", 0.0))
    )
    return round(total / loan_amount, 3) if loan_amount else 0.0


def risk_score(features: Dict[str, Any], config: PDModelConfig | None = None) -> Dict[str, Any]:
    pd_summary = probability_of_default(features, naics=str(features.get("industry_naics", "")), config=config)
    dscr = features.get("dscr")
    coverage = collateral_coverage(features.get("collateral", {}), features.get("loan_amount", 0.0))
    pd_summary["collateral_coverage"] = coverage
    pd_summary["dscr"] = dscr
    return pd_summary


def load_rules(path: Path | None = None) -> Dict[str, Any]:
    path = path or RULES_PATH
    if not path.exists():
        raise MissingArtifactError(f"Rules file missing at {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def policy_check(program: str, facts: Dict[str, Any], metrics: Dict[str, Any], path: Path | None = None) -> Dict[str, Any]:
    rules = load_rules(path)
    program_rules = rules.get("programs", {}).get(program, {})
    failed = []
    passes = True

    for rule in program_rules.get("rules", []):
        expr = rule.get("expr")
        if not expr:
            continue
        try:
            context = {"facts": facts, "metrics": metrics, "math": math}
            outcome = bool(eval(expr, {"__builtins__": {}}, context))
        except Exception:
            outcome = False
        if not outcome:
            passes = False
            failed.append({"id": rule.get("id"), "reason": rule.get("reason")})

    decision_table = program_rules.get("decision_table", [])
    decision = "undetermined"
    conditions: List[str] = []
    for row in decision_table:
        threshold_expr = row.get("when")
        context = {"facts": facts, "metrics": metrics, "math": math}
        if threshold_expr and not bool(eval(threshold_expr, {"__builtins__": {}}, context)):
            continue
        decision = row.get("decision", decision)
        conditions = row.get("conditions", [])
        break

    return {
        "passes": passes,
        "failed_rules": failed,
        "decision": decision,
        "conditions": conditions,
    }


def generate_letter(letter_type: str, bullets: Sequence[str], recipient: str) -> Dict[str, str]:
    greeting = "Approval" if letter_type == "approval" else "Adverse Action"
    body = "\n".join(f"• {b}" for b in bullets)
    draft = (
        f"{greeting} Letter\n\n"
        f"Recipient: {recipient}\n\n"
        f"Summary:\n{body}\n\n"
        "Please contact us if you have questions about this decision."
    )
    return {"draft": draft}


def compute_decision(program: str, features: Dict[str, Any], metrics: Dict[str, Any], rules_path: Path | None = None) -> Dict[str, Any]:
    pd_summary = risk_score(features)
    combined_metrics = {
        **metrics,
        "pd": pd_summary.get("pd"),
        "lgd": pd_summary.get("lgd"),
        "risk_band": pd_summary.get("risk_band"),
        "collateral_coverage": pd_summary.get("collateral_coverage"),
        "dscr": pd_summary.get("dscr") or metrics.get("dscr"),
    }
    policy = policy_check(program, features, combined_metrics, rules_path)
    return {
        "pd_summary": pd_summary,
        "policy": policy,
        "metrics": combined_metrics,
        "features": features,
    }


def underwrite_case(
    program: str,
    features: Dict[str, Any],
    loan_terms: Optional[Dict[str, Any]] = None,
    transactions_csv: Optional[str] = None,
    citation_query: Optional[str] = None,
    recipient: str | None = None,
    finalize: bool = False,
) -> Dict[str, Any]:
    loan_terms = loan_terms or {}
    timestamp = time.time()

    metrics: Dict[str, Any] = {}
    cashflow: Optional[Dict[str, Any]] = None
    if transactions_csv:
        cashflow = calc_cashflow(transactions_csv, loan_terms)
        metrics.update({k: v for k, v in cashflow.items() if k != "monthly_cashflow"})
        metrics["monthly_cashflow"] = cashflow.get("monthly_cashflow")

    features = dict(features)
    if cashflow and "dscr" in cashflow:
        features["dscr"] = cashflow["dscr"]
    metrics.setdefault("dscr", features.get("dscr"))

    loan_amount = loan_terms.get("amount", features.get("loan_amount", 0.0))
    metrics["loan_amount"] = loan_amount
    features["loan_amount"] = loan_amount
    features.setdefault("collateral", {})

    coverage = collateral_coverage(features.get("collateral", {}), loan_amount)
    metrics["collateral_coverage"] = coverage

    pd_summary = risk_score(features)
    metrics.update(
        {
            "pd": pd_summary.get("pd"),
            "lgd": pd_summary.get("lgd"),
            "risk_band": pd_summary.get("risk_band"),
            "collateral_coverage": pd_summary.get("collateral_coverage"),
            "dscr": pd_summary.get("dscr") or metrics.get("dscr"),
        }
    )

    policy = policy_check(program, features, metrics)

    decision = policy.get("decision", "undetermined")
    letter_type = "approval" if decision in {"approve", "conditional"} else "adverse"
    bullets: List[str] = [
        f"Program: {program.upper()}",
        f"Requested amount: ${loan_amount:,.0f}",
        f"PD: {metrics.get('pd', 0.0) * 100:.1f}% (Band {metrics.get('risk_band', '?')})",
        f"DSCR: {metrics.get('dscr', 'n/a')}",
    ]
    if policy.get("conditions"):
        bullets.extend(policy["conditions"])

    letter = generate_letter(letter_type, bullets, recipient or "Borrower")

    cite_query = citation_query or f"SBA {program} underwriting requirements"
    hits = search_docs(cite_query, k=40)
    quality = quality_score(cite_query, hits[:5]) if hits else 0.0
    citations = [
        {
            "doc": hit.metadata.get("doc"),
            "title": hit.metadata.get("title"),
            "page": hit.metadata.get("page"),
            "score": hit.score,
            "text": hit.text,
        }
        for hit in hits[:3]
    ]

    audit_entry = log_audit(
        "underwrite_case",
        {
            "timestamp": timestamp,
            "metadata": {"program": program, "finalize": finalize},
            "input": {
                "program": program,
                "features": features,
                "loan_terms": loan_terms,
                "transactions_csv": transactions_csv,
            },
            "output": {
                "decision": decision,
                "policy": policy,
                "metrics": metrics,
                "pd_summary": pd_summary,
                "citations": citations,
            },
        },
    )

    return {
        "program": program,
        "decision": decision,
        "policy": policy,
        "risk": pd_summary,
        "metrics": metrics,
        "conditions": policy.get("conditions", []),
        "quality": quality,
        "citations": citations,
        "letter": letter,
        "cashflow": cashflow,
        "audit": {
            "tool": audit_entry.tool,
            "input_hash": audit_entry.input_hash,
            "output_hash": audit_entry.output_hash,
        },
    }

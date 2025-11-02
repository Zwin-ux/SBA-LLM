# Underwriter Copilot Architecture Overview

## Goals

- Accelerate SBA credit file reviews while maintaining consistent application of SOP and lender overlays.
- Provide transparent, citeable decisions with structured outputs (letters, memos, audit logs).
- Layer underwriting-specific tooling on top of the existing RAG stack so LLM agents operate with grounded data.

## End-to-End Loop

1. **Parse Intake** – Collect borrower package, run `extract_docs` to normalize statements, financials, and uploaded forms.
2. **Retrieve Policy** – Use existing FAISS index + reranker via `search_docs`/`rerank_hits` to fetch SOP references for the scenario.
3. **Compute Metrics** – Run analytical tools (`calc_cashflow`, `risk_score`, `eligibility_compare`) to derive DSCR, coverage, PD, program fit.
4. **Apply Rules** – Evaluate policy rules defined in `config/rules.yaml` using borrower facts and computed metrics through `policy_check`.
5. **Propose Decision** – Combine quantitative metrics + rule outcomes via decision table logic to recommend approve/conditional/decline.
6. **Generate Artifacts** – Produce narrative outputs (`generate_letter`, memo templates, conditions list) with citations referencing retrieved policy chunks.
7. **Log Audit** – Persist immutable JSON log with hashed inputs/outputs via `log_audit` to satisfy compliance requirements.

## Key Modules

- `agent_tools.py` (existing): retrieval, quality scoring, extractive fallback.
- `underwriter/tools.py`: wrappers for document extraction, cash-flow computation, risk scoring, policy checks, letter generation, citation building, and audit logging.
- `underwriter/pd_model.py`: transparent logistic regression (weights stored in `config/features.json`).
- `underwriter/rules.py`: YAML-based policy/rule evaluation + decision table execution.
- `agent_server.py`: expose new endpoints (`/underwrite`, `/policy_check`, `/calc_cashflow`, etc.) reusing shared logic.

## Config Artifacts

- `config/features.json`: feature definitions, coefficient weights, scaling guidance for PD model.
- `config/rules.yaml`: SOP + overlay rules, decision thresholds, condition templates.

## Audit & Guardrails

- Every tool returns provenance (document IDs, source hashes) to support cite-or-decline policy.
- Audit logs include SHA256 hashes of inputs/outputs for tamper detection.
- Deterministic mode toggled by request flag (`finalize=true`) to disable sampling in generation.

## UI Considerations

- **Left panel**: chat history + tool call trace.
- **Center card**: PD %, DSCR, decision (Approve/Conditional/Decline), conditions list, confidence.
- **Right panel**: expandable sources with PDF thumbnails and SOP citations.
- Buttons to regenerate decisions, request docs, download letters.

This design keeps the current RAG foundation, adds deterministic analytics, and prepares the path for incremental automation of underwriting workflows.

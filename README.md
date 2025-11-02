---
title: SBA-LLM
sdk: gradio
app_file: app.py
license: apache-2.0
emoji:
pinned: true
---

# SBA Copilot & Underwriter Assistant

## Executive Summary

SBA Copilot is a knowledge and workflow assistant that helps Small Business Administration (SBA) teams and lending partners answer policy questions, review borrower files, and draft communications with confidence. The system blends trusted SBA source material with deterministic analytics so every recommendation comes with citations, risk metrics, and an auditable trail.

- **For leadership:** faster credit decisions, fewer manual lookups, and explainable outputs that are easy to review.
- **For underwriters and analysts:** retrieval-augmented answers, automated policy checks, and ready-to-send letters and memos.
- **For borrowers and outreach teams:** consistent guidance, quicker feedback on eligibility, and traceable decisions tied to SBA rules.

## High-Level Concept

1. **Grounding in SBA sources** – PDFs and HTML guidance from sba.gov and partner lenders are indexed with vector search so the assistant always cites official policy.
2. **Agentic workflow** – An orchestration layer calls structured tools (search, cash-flow analysis, policy check, letter generation) before synthesizing answers.
3. **Explainable outcomes** – Every decision surfaces PD/DSCR metrics, failing rules, compensating factors, and links back to the page or SOP section that justifies it.

## System Components (Technical View)

- **RAG Engine (`rag/ingest.py`, `rag/index.faiss`, `rag/meta.jsonl`)**
  - Embeddings via `thenlper/gte-large`
  - FAISS flat index for fast retrieval
  - Manifest-driven metadata (doc IDs, pages, scope)
- **Streamlit Demo (`app.py`)**
  - Q&A UI with latency + quality metrics
  - LLM pipeline: retrieve → rerank → synthesize → cite
  - Supports OpenAI (if key provided) or open-weight Qwen3 generator
- **Agent Toolkit (`agent_tools.py`)**
  - Shared functions for search, rerank, quality scoring, email draft, CRM logging
  - Exposed through FastAPI for downstream integrations
- **Underwriter Toolkit (`underwriter/tools.py`, `underwriter/pd_model.py`)**
  - Cash-flow calculator (DSCR, EBITDA, debt service)
  - Logistic regression PD/LGD scoring with explainable contributions
  - Policy rules + decision table execution (`config/rules.yaml`)
  - Decision composer that generates letters, citations, audit hashes
- **Agent API (`agent_server.py`)**
  - Endpoints: `/answer`, `/search`, `/calc_cashflow`, `/risk_score`, `/policy_check`, `/underwrite`, `/draft_email`, `/log_note`
  - Ready for integration with LOS/CRM, or to power the Streamlit frontend

## How an Underwriting Session Works

1. **Parse inputs** – Borrower uploads are normalized (placeholder `extract_docs` today, ready for OCR/Table extraction).
2. **Retrieve policy** – FAISS lookup surfaces relevant SOP excerpts.
3. **Compute metrics** – Cash-flow CSVs produce DSCR & revenue; collateral inputs yield coverage.
4. **Score risk** – Logistic PD model assigns risk band with SHAP-like contribution text.
5. **Apply rules** – YAML rules evaluate eligibility and overlays (e.g., DSCR ≥ 1.20 for 7(a)).
6. **Draft outputs** – Agent proposes decision, conditions, approval/adverse letter, and cites sources.
7. **Audit & log** – Inputs/outputs hashed and appended to JSONL audit trail (`logs/audit`).

## Quick Start (Tech Checklist)

```bash
# 1. Install dependencies (Streamlit, FAISS, FastAPI, pandas, torch, etc.)
pip install -r requirements.txt

# 2. Build the knowledge index from SBA documents
python rag/ingest.py

# 3. Run the Streamlit boss demo
python -m streamlit run app.py

# 4. Launch the agent backend (for API or future frontend integrations)
uvicorn agent_server:app --reload

# 5. Evaluate retrieval quality (Hit@5, MRR)
python eval.py
```

> **Tip:** Set `QWEN_MODEL_NAME` if you want a different open-weight model. Provide `OPENAI_API_KEY` to switch the answer generator back to GPT.

## Tools & APIs Exposed

| Endpoint / Tool | Purpose | Sample Use |
| --- | --- | --- |
| `search_docs` / `/search` | Pull top SBA snippets with metadata | Frontend citation cards |
| `calc_cashflow` | Compute monthly cashflow, DSCR, debt service | Bank statement ingestion |
| `risk_score` | Logistic PD/LGD + per-feature contributions | Decision card + exception memos |
| `policy_check` | Run SOP + overlay rules with decisions/conditions | Eligibility triage |
| `underwrite_case` | Full loop: metrics, rules, letter, audit hashes | Agent “Underwriter Copilot” action |
| `generate_letter` | Approval or adverse action draft | Downloadable PDFs (hook into e-sign) |
| `log_audit` | Append tamper-evident JSON trace | Compliance evidence |

## Governance & Guardrails

- **Cite-or-decline**: All claims must reference a source chunk; otherwise explicitly marked as assumptions.
- **Deterministic finalization**: When `finalize=true`, sampling is disabled and outputs are stored with hashes.
- **PII handling**: Document extraction layer can mask SSNs and store encrypted blobs (stubs ready for integration).
- **Policy drift monitoring**: Re-ingest SOPs on schedule; `config/rules.yaml` allows quick updates when SBA releases changes.

## Roadmap Suggestions

1. **Upgrade `extract_docs`** with OCR/table extraction for bank statements and tax returns.
2. **Connect to live LOS/CRM** via `/underwrite` + `/log_note` routes.
3. **Add change detection**: nightly diff between SOP versions and alert underwriters when rules shift.
4. **Scenario testing**: run scripted borrower cases, capture latency + accuracy dashboards for leadership.
5. **UI polish**: build the three-panel “decision cockpit” (chat, decision card, sources) described in the design brief.

## Repository Map

- `rag/` – Data ingestion scripts and source documents
- `app.py` – Streamlit demo showcasing Q&A with citations and metrics
- `agent_tools.py` – Shared retrieval/answer utilities for both UI and backend
- `underwriter/` – Analytical models, rules evaluation, underwriting workflow helpers
- `agent_server.py` – FastAPI service exposing agent and underwriting endpoints
- `config/` – PD coefficients and policy rules (editable without touching code)
- `docs/underwriter_design.md` – Architecture note for engineering team

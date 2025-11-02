# WindSurf Project Prompt: Salesforce SBA Copilot & Credit Analyzer Agents

## Project Overview

You are building two coordinated services that embed the SBA Copilot intelligence into Salesforce:

1. **sba_copilot_salesforce** – a retrieval + policy reasoning agent that lives in the lender’s Salesforce console. It reads opportunity data, checks SBA policy, drafts communications, and logs auditable recommendations.
2. **credit_analyzer_agent** – a quantitative risk analysis companion that extracts borrower financials, computes DSCR/PD/LGD, and suggests conditions. Outputs are pushed into Salesforce custom objects and dashboards.

Both services use the existing Python FastAPI backend (vector retrieval, underwriting tools, PD model) and expose Salesforce-friendly REST endpoints. The deliverable includes backend routes, Lightning Web Component (LWC) stubs, Apex callouts, and schema definitions for auditing.

## Top-Level Structure

```text
/ (root)
├── backend/
│   ├── main.py               # FastAPI app w/ new endpoints under /api/
│   ├── routers/
│   │   ├── copilot.py        # Eligibility + policy routes
│   │   ├── analyzer.py       # Cashflow + risk scoring routes
│   │   └── health.py
│   ├── services/
│   │   ├── copilot_service.py
│   │   ├── analyzer_service.py
│   │   └── salesforce_sync.py
│   ├── models/
│   │   ├── request.py        # Pydantic models for requests/responses
│   │   └── schemas.py        # Shared DTOs, audit records
│   └── config.py             # env settings (API keys, Salesforce creds)
├── salesforce/
│   ├── lwc/
│   │   ├── copilotPanel/
│   │   │   ├── copilotPanel.html
│   │   │   ├── copilotPanel.js
│   │   │   └── copilotPanel.css
│   │   ├── riskSummary/
│   │   │   ├── riskSummary.html
│   │   │   └── riskSummary.js
│   ├── apex/
│   │   ├── CopilotController.cls
│   │   ├── CreditAnalyzerController.cls
│   │   └── CopilotAuditService.cls
│   └── objects/
│       ├── Copilot_Audit__c.object-meta.xml
│       └── Credit_Summary__c.object-meta.xml
├── infra/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── github-actions.yml
└── docs/
    ├── integration_guide.md
    ├── salesforce_setup.md
    └── api_reference.md
```

## Backend Requirements

- Reuse logic from `agent_tools` and `underwriter.tools` through service layer wrappers.
- Configure routes under `/api/sba-copilot/*` and `/api/credit-analyzer/*`.
- Provide JSON responses optimized for Salesforce (flat keys, stringified IDs).
- Add audit logging (hashes + timestamps) for every decision.
- Support `finalize` flag to run deterministic flows.
- Include unit tests for services and routers.

### Example Endpoints

```http
POST /api/sba-copilot/check
Body: {
  "opportunity_id": "006xx",
  "program": "7a",
  "facts": {"citizenship": "US", "use_of_proceeds": "working_capital"},
  "metrics": {"dscr": 1.24, "loan_amount": 350000}
}
Response: {
  "opportunity_id": "006xx",
  "eligibility": true,
  "confidence": 0.91,
  "failed_rules": [],
  "summary": "Meets 7(a) eligibility requirements.",
  "citations": ["SOP50-10-7-§3.2"],
  "letter_draft": "...",
  "audit": {"input_hash": "...", "output_hash": "..."}
}
```

```http
POST /api/credit-analyzer/score
Body: {
  "loan_id": "LN-2345",
  "industry_naics": "722513",
  "loan_amount": 550000,
  "transactions_csv": "s3://.../bank.csv",
  "collateral": {"real_estate": 400000, "equipment": 120000}
}
Response: {
  "loan_id": "LN-2345",
  "dscr": 1.28,
  "pd": 0.057,
  "lgd": 0.41,
  "risk_band": "B",
  "recommended_conditions": ["Provide 2023 tax return", "Maintain DSCR ≥ 1.1"],
  "citations": ["SOP50-10-504-§5"],
  "audit": {...}
}
```

## Salesforce Deliverables

1. **Lightning Web Components**
   - `copilotPanel` renders chat, decision card, and action buttons (“Insert into Notes”, “Generate Letter”).
   - `riskSummary` shows DSCR, PD, risk band, and conditions on the Opportunity or Loan object.
   - Use Lightning data service to fetch record fields; call Apex methods to reach the backend.

2. **Apex Controllers**
   - `CopilotController` – wraps HTTP callouts to `/api/sba-copilot/check`, handles Named Credential, stores audit results in `Copilot_Audit__c`.
   - `CreditAnalyzerController` – triggers scoring, saves metrics into `Credit_Summary__c` object.
   - `CopilotAuditService` – utility for logging outcomes, handling retries.

3. **Custom Objects & Fields**
   - `Copilot_Audit__c`: fields `Opportunity__c (Lookup)`, `Decision__c`, `Confidence__c`, `Citations__c`, `Notes__c`, `Input_Hash__c`, `Output_Hash__c`, `CreatedByCopilot__c`.
   - `Credit_Summary__c`: fields `Loan__c (Lookup)`, `DSCR__c`, `PD__c`, `Risk_Band__c`, `Conditions__c`, `Last_Rescore__c`.

4. **Flows / Triggers**
   - Flow button “Check Eligibility” → Apex call → update `Copilot_Audit__c` and show toast.
   - Scheduled Apex (or Flow) to run nightly `/api/credit-analyzer/score` for active loans.

## Deployment Scripts

- Docker image running FastAPI + gunicorn. Include env vars for Salesforce credentials, OpenAI, optional Qwen endpoints.
- GitHub Actions workflow to build/push container and deploy to Render/Fly.io/AWS.
- Salesforce DX or Metadata API script to deploy LWCs and objects.

## Docs to Produce

- `integration_guide.md`: step-by-step for connecting Salesforce org, setting Named Credentials, testing callouts.
- `salesforce_setup.md`: object definitions, permission sets, layout changes.
- `api_reference.md`: describe each REST endpoint, request/response schema, auth headers.

## Stretch Goals

- Add webhook from FastAPI when new SOP is ingested (Slack notification for policy drift).
- Real-time streaming (Platform Events) so Salesforce users see updates instantly.
- Portfolio monitoring dashboard (Tableau/Einstein) reading from `Credit_Summary__c`.

## General Guidance for WindSurf Project

- Follow existing code style (`black`, `ruff` friendly). Provide tests.
- Reuse existing PD model/underwriter utilities via service layer rather than duplicating logic.
- Ensure every decision includes citations and hashed audit records.
- Keep prompts/config in environment variables; no hard-coded secrets.
- Document how to run end-to-end demo (API + Salesforce UI) in `docs/integration_guide.md`.

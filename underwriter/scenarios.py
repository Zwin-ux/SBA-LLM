"""Reusable underwriting scenarios for demos."""

from __future__ import annotations

SCENARIOS = {
    "Custom question": {
        "question": "What are eligibility rules for SBA 504 loans?",
        "description": "Ask any SBA policy or program question and explore cited answers.",
    },
    "504 Manufacturing Expansion": {
        "question": "Underwrite a 504 manufacturing expansion loan and list required conditions.",
        "program": "504",
        "features": {
            "industry_naics": "333120",
            "years_in_business": 8,
            "rev_ttm": 2_450_000,
            "dscr": 1.32,
            "dti": 0.28,
            "fico_proxy": 710,
            "delinquencies": False,
            "collateral": {"real_estate": 850_000, "equipment": 250_000},
        },
        "loan_terms": {"amount": 750_000, "rate": 0.065, "term_months": 240},
        "citation_query": "SBA 504 manufacturing eligibility requirements",
        "description": "Existing manufacturer expanding production capacity via a 504 loan.",
    },
    "7(a) Working Capital": {
        "question": "What conditions are required for a 7(a) café with DSCR 1.08?",
        "program": "7a",
        "features": {
            "industry_naics": "722513",
            "years_in_business": 6,
            "rev_ttm": 1_200_000,
            "dscr": 1.08,
            "dti": 0.34,
            "fico_proxy": 690,
            "delinquencies": False,
            "collateral": {"real_estate": 350_000, "equipment": 90_000},
        },
        "loan_terms": {"amount": 350_000, "rate": 0.11, "term_months": 120},
        "citation_query": "SBA 7(a) underwriting DSCR requirements",
        "description": "Neighborhood café seeking working-capital financing under 7(a).",
    },
}

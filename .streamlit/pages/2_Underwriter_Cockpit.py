import json
from datetime import datetime
from typing import Any, Dict

import pandas as pd
import streamlit as st

from underwriter.scenarios import SCENARIOS
from underwriter.tools import underwrite_case


def render_decision_card(result: Dict[str, Any]) -> None:
    cols = st.columns(4)
    cols[0].metric("Decision", result.get("decision", "n/a").title())
    cols[1].metric("Risk band", result.get("risk", {}).get("risk_band", "-"))
    cols[2].metric("PD", f"{result.get('risk', {}).get('pd', 0)*100:.2f}%")
    cols[3].metric("DSCR", f"{result.get('metrics', {}).get('dscr', 0):.2f}")

    st.caption(
        f"Confidence score: {result.get('quality', 0):.3f} | Loan amount: ${result.get('metrics', {}).get('loan_amount', 0):,.0f}"
    )

    if result.get("conditions"):
        st.subheader("Conditions")
        st.info("\n".join(f"• {condition}" for condition in result["conditions"]))
    else:
        st.success("No additional conditions required.")

    policy = result.get("policy", {})
    failed_rules = policy.get("failed_rules", [])
    if failed_rules:
        st.subheader("Failed rules")
        for rule in failed_rules:
            st.error(f"{rule.get('id')}: {rule.get('reason')}")

    contrib = result.get("risk", {}).get("contributions", {})
    if contrib:
        st.subheader("Feature contributions")
        contrib_df = pd.DataFrame(
            [(feature, value) for feature, value in contrib.items()],
            columns=["Feature", "Contribution (log-odds)"],
        ).sort_values("Contribution (log-odds)", ascending=False)
        st.dataframe(contrib_df, use_container_width=True)

    citations = result.get("citations", [])
    if citations:
        st.subheader("Citations")
        for idx, cite in enumerate(citations, start=1):
            with st.expander(f"[{idx}] {cite.get('title', cite.get('doc'))} • p.{cite.get('page', '?')}"):
                st.write(cite.get("text", ""))

    st.subheader("Audit trail")
    st.code(json.dumps(result.get("audit", {}), indent=2))


st.set_page_config(page_title="Underwriter Cockpit", layout="wide")
st.title("Underwriter Cockpit")
st.caption("Real-time SBA underwriting decisions with citations, risk metrics, and audit trail.")

with st.sidebar:
    st.header("Scenario")
    scenario_name = st.selectbox("Choose scenario", list(SCENARIOS.keys()), index=0)
    preset = SCENARIOS[scenario_name]

    program = st.selectbox("SBA Program", ["7a", "504", "microloan"], index=["7a", "504", "microloan"].index(preset.get("program", "7a")))

    amount = st.number_input(
        "Requested amount",
        value=float(preset.get("loan_terms", {}).get("amount", 250000)),
        step=50_000.0,
        min_value=50_000.0,
    )

    rate = st.number_input(
        "Interest rate (decimal)",
        value=float(preset.get("loan_terms", {}).get("rate", 0.1)),
        step=0.005,
        format="%.3f",
    )

    term_months = st.number_input(
        "Term (months)",
        value=int(preset.get("loan_terms", {}).get("term_months", 120)),
        step=12,
        min_value=12,
    )

    dscr = st.number_input("DSCR", value=float(preset.get("features", {}).get("dscr", 1.2)), step=0.05)
    fico = st.number_input("FICO proxy", value=float(preset.get("features", {}).get("fico_proxy", 700)), step=5.0)
    dti = st.number_input("Debt-to-income", value=float(preset.get("features", {}).get("dti", 0.3)), step=0.01, format="%.2f")
    years = st.number_input("Years in business", value=float(preset.get("features", {}).get("years_in_business", 5)), step=1.0)

    finalize = st.checkbox("Finalize (deterministic output)", value=False)

    run_button = st.button("Run Underwrite")

main_col, sources_col = st.columns([2, 1])

if run_button:
    with st.spinner("Running underwriting pipeline..."):
        payload = {
            "program": program,
            "features": {
                **preset.get("features", {}),
                "dscr": dscr,
                "fico_proxy": fico,
                "dti": dti,
                "years_in_business": years,
                "loan_amount": amount,
            },
            "loan_terms": {"amount": amount, "rate": rate, "term_months": term_months},
            "citation_query": preset.get("citation_query"),
            "finalize": finalize,
        }

        try:
            result = underwrite_case(
                program=payload["program"],
                features=payload["features"],
                loan_terms=payload["loan_terms"],
                citation_query=payload.get("citation_query"),
                finalize=payload.get("finalize", False),
            )
        except Exception as exc:
            st.error(f"Underwriting failed: {exc}")
            st.stop()

    with main_col:
        render_decision_card(result)

    with sources_col:
        st.subheader("Snapshot")
        st.markdown(
            f"**Timestamp:** {datetime.utcnow().isoformat()}Z\n\n"
            f"**Quality:** {result.get('quality', 0):.3f}\n\n"
            f"**Program:** {result.get('program', '').upper()}"
        )
        st.subheader("Monthly Cashflow")
        cashflow = result.get("cashflow", {}).get("monthly_cashflow", {})
        if cashflow:
            cash_df = pd.DataFrame.from_dict(cashflow, orient="index")
            cash_df.index.name = "Month"
            st.dataframe(cash_df, use_container_width=True)
        else:
            st.write("No cashflow data provided.")

        st.subheader("Sources")
        for idx, cite in enumerate(result.get("citations", []), start=1):
            st.markdown(
                f"[{idx}] **{cite.get('title', cite.get('doc'))}**\n"
                f"Score: {cite.get('score', 0):.3f} • Page {cite.get('page', '?')}"
            )
else:
    st.info("Configure the scenario in the sidebar and click **Run Underwrite**.")

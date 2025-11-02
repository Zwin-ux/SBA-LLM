import os, json, time, pathlib
from copy import deepcopy
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from rapidfuzz import fuzz
import streamlit as st
from underwriter.scenarios import SCENARIOS
from underwriter.tools import underwrite_case

# Optional local generator (Qwen3) dependencies are loaded lazily
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # pragma: no cover - handled at runtime if transformers not available
    AutoModelForCausalLM = AutoTokenizer = None

DATA_DIR = pathlib.Path("rag/data")
IDX_PATH  = "rag/index.faiss"
META_PATH = "rag/meta.jsonl"
QWEN_MODEL_NAME = os.getenv("QWEN_MODEL_NAME", "Qwen/Qwen3-7B-Instruct")

# ---- Load artifacts
@st.cache_resource
def load_resources():
    emb = SentenceTransformer("thenlper/gte-large")
    index = faiss.read_index(str(IDX_PATH))
    meta = [json.loads(l) for l in open(META_PATH, "r", encoding="utf-8")]
    try:
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    except Exception:
        reranker = None
    return emb, index, meta, reranker

emb, index, meta, reranker = load_resources()

@st.cache_resource
def load_qwen_generator(model_name: str = QWEN_MODEL_NAME):
    """Load a local Qwen3 chat model if transformers/torch are available."""
    if AutoTokenizer is None or AutoModelForCausalLM is None:
        raise RuntimeError("transformers not installed; cannot load Qwen model")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - runtime check
        raise RuntimeError("PyTorch not installed; install torch to use Qwen") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    )

    def generate(prompt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": "You are an SBA policy expert. Answer using only provided context and cite like [1],[2].",
            },
            {"role": "user", "content": prompt},
        ]
        input_ids = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(model.device)
        gen_ids = model.generate(
            input_ids,
            max_new_tokens=600,
            temperature=0.2,
            do_sample=False,
        )
        output = tokenizer.decode(gen_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        return output.strip()

    return generate

def retrieve(query, k=40):
    qv = emb.encode([query], normalize_embeddings=True)
    D, I = index.search(np.asarray(qv, dtype="float32"), k)
    # convert FAISS L2 on normalized vecs → cosine similarity proxy
    sims = (1 - D[0] / 2.0).tolist()
    hits = [{"i": int(i), "score": float(sims[idx]), **meta[i]} for idx, i in enumerate(I[0])]
    return hits

def rerank(query, hits, topn=6):
    if not hits: return []
    if reranker is None or len(hits) <= topn:
        return sorted(hits, key=lambda h: -h["score"])[:topn]
    pairs = [(query, h["text"]) for h in hits]
    rr = reranker.predict(pairs)
    for h, s in zip(hits, rr): h["rr_score"] = float(s)
    hits.sort(key=lambda h: -h.get("rr_score", h["score"]))
    return hits[:topn]

def synthesize_answer(query, ctx_chunks):
    """If OPENAI_API_KEY is set, call OpenAI; else extractive summary."""
    ctx = "\n\n".join(f"[{k+1}] {c['text']}" for k, c in enumerate(ctx_chunks))
    if os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            client = OpenAI()
            prompt = (
                "Answer using only the context. Cite sources like [1],[2]."
                "\n\nContext:\n" + ctx + f"\n\nQuestion: {query}\nAnswer:"
            )
            msg = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"user","content":prompt}],
                temperature=0.2,
            )
            return msg.choices[0].message.content.strip()
        except Exception:
            pass
    # Try local Qwen3 generator
    try:
        generator = load_qwen_generator()
        prompt = (
            "Answer using only the context. Cite sources like [1],[2]."
            "\n\nContext:\n" + ctx + f"\n\nQuestion: {query}\nAnswer:"
        )
        return generator(prompt)
    except Exception:
        pass
    # Fallback: extractive. Pick the best chunk by fuzzy match and return a trimmed answer.
    best = max(ctx_chunks, key=lambda c: fuzz.partial_ratio(query, c["text"]))
    txt = best["text"]
    # crude trim to ~750 chars around best substring
    pos = txt.lower().find(query.split()[0].lower()) if query.split() else 0
    start = max(0, pos - 350)
    end = min(len(txt), start + 750)
    return f"{txt[start:end].strip()}  \n\nSources: " + " ".join(f"[{k+1}]" for k,_ in enumerate(ctx_chunks,1))

def quality_score(query, ctx_chunks):
    # Combine normalized similarity, reranker if present, and coverage proxy
    sim = np.mean([c.get("rr_score", c["score"]) for c in ctx_chunks]) if ctx_chunks else 0.0
    coverage = max([fuzz.token_set_ratio(query, c["text"]) for c in ctx_chunks] + [0]) / 100.0
    return round(0.6*sim + 0.4*coverage, 3)

# ---- UI
st.set_page_config(page_title="SBA Copilot", layout="wide")
st.title("SBA Copilot")
st.caption(
    "AI knowledge & workflow assistant for SBA teams — cited answers, latency tracking, and quality scoring."
)

with st.expander("How it works", expanded=False):
    st.markdown(
        "1. **Retrieve** relevant policy snippets via FAISS.\n"
        "2. **Rerank** with cross-encoder for precise matches.\n"
        "3. **Answer** with citations, then compute a blended confidence score."
    )

with st.sidebar:
    st.header("Retrieval Settings")
    k = st.slider("Top-K retrieve", 10, 100, 40, step=10)
    topn = st.slider("Top-N rerank", 3, 10, 6)
    st.caption("Tune to explore recall vs. precision.")
    st.divider()
    st.markdown("**Scenario bundles**")
    scenario_name = st.selectbox("Choose a scenario", list(SCENARIOS.keys()), index=0)
    scenario_data = deepcopy(SCENARIOS.get(scenario_name, {}))
    if scenario_data.get("description"):
        st.caption(scenario_data["description"])

    default_question = scenario_data.get("question", "What are eligibility rules for SBA 504 loans?")

q = st.text_input(
    "Ask a question",
    value=default_question,
    placeholder="e.g., What are eligibility rules for 504 loans?",
)

show_explain = st.toggle(
    "Show underwriting explanation",
    value=bool(scenario_data.get("features")),
    help="Overlay the underwriting engine to surface PD contributions, failed rules, and audit hashes.",
)

if q:
    t0 = time.time()
    hits = retrieve(q, k=k)
    top = rerank(q, hits, topn=topn)
    ans = synthesize_answer(q, top)
    t_ms = int((time.time() - t0)*1000)
    score = quality_score(q, top)

    st.subheader("Answer")
    st.write(ans)
    col1, col2, col3 = st.columns(3)
    col1.metric("Latency", f"{t_ms} ms")
    col2.metric("Quality score", f"{score}")
    col3.metric("Context chunks", f"{len(top)}")

    st.subheader("Sources")
    for idx, h in enumerate(top, start=1):
        with st.expander(f"[{idx}] {h.get('doc_id','')}  |  score={round(h.get('rr_score',h['score']),3)}"):
            st.write(h["text"])
            # Optional page preview if your meta stored it:
            # st.caption(f"{h.get('source_path','')}  p.{h.get('page', '?')}")

    if scenario_data.get("features") and show_explain:
        st.subheader("Underwriting decision insights")
        try:
            uw_result = underwrite_case(
                program=scenario_data.get("program", "7a"),
                features=scenario_data["features"],
                loan_terms=scenario_data.get("loan_terms"),
                citation_query=scenario_data.get("citation_query"),
                finalize=False,
            )
            metrics = uw_result.get("metrics", {})
            policy = uw_result.get("policy", {})
            risk = uw_result.get("risk", {})
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("Decision", uw_result.get("decision", "n/a").title())
            col_b.metric("Risk band", risk.get("risk_band", "-"))
            col_c.metric("PD", f"{risk.get('pd', 0.0)*100:.2f}%")
            col_d.metric("DSCR", f"{metrics.get('dscr', 0.0):.2f}")

            conditions = policy.get("conditions", [])
            if conditions:
                st.info("\n".join(f"• {c}" for c in conditions))
            else:
                st.success("No additional conditions required.")

            failed_rules = policy.get("failed_rules", [])
            if failed_rules:
                st.error("Failed rules:" + "\n" + "\n".join(f"- {r.get('id')}: {r.get('reason')}" for r in failed_rules))

            contrib = risk.get("contributions", {})
            if contrib:
                contrib_df = pd.DataFrame(
                    [(k, v) for k, v in contrib.items()],
                    columns=["Feature", "Contribution"],
                ).sort_values("Contribution", ascending=False)
                st.markdown("**Feature contributions (log-odds)**")
                st.dataframe(contrib_df, use_container_width=True)

            citations = uw_result.get("citations", [])
            if citations:
                with st.expander("Citations"):
                    for idx, cite in enumerate(citations, start=1):
                        st.markdown(
                            f"**[{idx}] {cite.get('title', cite.get('doc'))}** — p.{cite.get('page', '?')}\n\n{cite.get('text', '')}"
                        )

            with st.expander("Audit record"):
                st.code(json.dumps(uw_result.get("audit", {}), indent=2))
        except Exception as exc:  # pragma: no cover - display errors in UI
            st.warning(f"Unable to generate underwriting insights: {exc}")

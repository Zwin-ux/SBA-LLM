# rag/ingest.py
from __future__ import annotations
from pypdf import PdfReader
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss, json, numpy as np, re, pathlib, os, datetime as dt
from typing import Dict, List

DATA_DIR = pathlib.Path("rag/data")
IDX_PATH = "rag/index.faiss"
META_PATH = "rag/meta.jsonl"
MANIFEST = {m["file"]: m for m in (
    [json.loads(x) for x in open(DATA_DIR / "manifest.jsonl", "r", encoding="utf-8").read().splitlines()]
    if (DATA_DIR / "manifest.jsonl").exists() else []
)}

EMB = SentenceTransformer("thenlper/gte-large")  # solid general embedder

def clean(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "")).strip()

def chunk_text(txt: str, doc_id: str, base_meta: Dict, page_hint: int | None = None) -> List[Dict]:
    # sentence-ish splits into ~800–1200 token chunks; keep simple
    parts = re.split(r"(?<=\.)\s{2,}", txt)
    out = []
    for i, part in enumerate(parts):
        if len(part) < 200:
            continue
        meta = {
            "doc": doc_id,
            "page": page_hint or (i + 1),
            "text": part[:6000],
            "id": f"{doc_id}_{page_hint or 0}_{abs(hash(part))%10**9}",
        }
        meta.update(base_meta)
        # Light boost marker for 504 content
        meta["boost_504"] = bool(re.search(r"\b504\b|Section\s*C\b|CDC", part, re.I))
        out.append(meta)
    return out

def pdf_to_sections(path: pathlib.Path, doc_id: str, base_meta: Dict) -> List[Dict]:
    r = PdfReader(str(path))
    out = []
    for i, page in enumerate(r.pages):
        txt = clean(page.extract_text() or "")
        if not txt:
            continue
        out.extend(chunk_text(txt, doc_id, base_meta, page_hint=i+1))
    return out

def html_to_sections(path: pathlib.Path, doc_id: str, base_meta: Dict) -> List[Dict]:
    html = open(path, "r", encoding="utf-8", errors="ignore").read()
    soup = BeautifulSoup(html, "lxml")
    # drop nav/scripts
    for tag in soup(["script", "style", "nav", "footer", "header"]): tag.decompose()
    text = clean(soup.get_text(" "))
    return chunk_text(text, doc_id, base_meta, page_hint=None)

def parse_effective(manifest_entry: Dict) -> str:
    ef = manifest_entry.get("effective") if manifest_entry else None
    try:
        return str(dt.date.fromisoformat(ef)) if ef else "unknown"
    except Exception:
        return "unknown"

def build_index():
    metas, vecs = [], []
    files = sorted(DATA_DIR.glob("*.*"))
    for fp in files:
        if fp.name == "manifest.jsonl":
            continue
        doc_id = fp.stem
        m = MANIFEST.get(fp.name, {})
        base_meta = {
            "file": fp.name,
            "title": m.get("title", doc_id),
            "url": m.get("url", ""),
            "effective": parse_effective(m),
            "scope": m.get("scope", "unknown"),
        }
        if fp.suffix.lower() in [".pdf"]:
            secs = pdf_to_sections(fp, doc_id, base_meta)
        elif fp.suffix.lower() in [".html", ".htm"]:
            secs = html_to_sections(fp, doc_id, base_meta)
        else:
            continue
        if not secs: 
            continue
        embs = EMB.encode([s["text"] for s in secs], normalize_embeddings=True, batch_size=64)
        for s, v in zip(secs, embs):
            metas.append(s); vecs.append(np.asarray(v, dtype="float32"))
    if not vecs:
        raise SystemExit("No chunks produced. Check rag/data files.")
    X = np.vstack(vecs)
    # Inner product index
    idx = faiss.IndexFlatIP(X.shape[1]); idx.add(X)
    faiss.write_index(idx, IDX_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        for m in metas: f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"wrote {len(metas)} chunks -> {IDX_PATH}, {META_PATH}")

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    build_index()

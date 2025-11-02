from typing import Any, Dict
from pypdf import PdfReader

def analyze(file) -> Dict[str, Any]:
    """
    Basic PDF analyzer stub: reads text from the uploaded PDF and returns a JSON
    with placeholders for eligibility notes and risk flags.

    Input can be a file-like object or a path as provided by Gradio's File input.
    """
    # Gradio's File provides a tempfile.NamedTemporaryFile-like object
    path = getattr(file, "name", None) or str(file)
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    full_text = "\n\n".join(pages)
    return {
        "summary": full_text[:2000],  # preview first 2k chars
        "pages": len(reader.pages),
        "eligibility_notes": [
            "Stub: parse SOP references and eligibility criteria from PDF text."
        ],
        "risk_flags": [],
    }

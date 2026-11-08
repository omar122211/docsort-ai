"""DocSort AI web page.  Run the API first, then:  streamlit run ui/streamlit_app.py

The page talks to the FastAPI backend (API_URL) and never sees the Gemini key.
"""
from __future__ import annotations

import os
from pathlib import Path

import httpx
import streamlit as st

from docsort.classes import LABELS
from docsort.report import build_report
from docsort.service import Result

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")
SAMPLES = Path(__file__).resolve().parent.parent / "samples"
ICONS = {"email": "📧", "form": "📝", "invoice": "🧾", "letter": "✉️", "receipt": "🛒", "resume": "👤", "scientific_publication": "🔬"}

st.set_page_config(page_title="DocSort AI", page_icon="🗂️", layout="wide")


def show(value) -> str:
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return f"{value:,}" if isinstance(value, (int, float)) else str(value)


def api(method: str, path: str, **kw):
    try:
        r = httpx.request(method, f"{API_URL}{path}", timeout=120, **kw)
    except httpx.HTTPError:
        return None, "The DocSort service is not reachable. Is the API running?"
    if r.status_code != 200:
        try:
            detail = r.json().get("detail")
        except ValueError:
            detail = None
        return None, detail if isinstance(detail, str) else "Something went wrong — please try again."
    return r.json(), None


# ---------------------------------------------------------------- sidebar
health, _ = api("GET", "/health")
with st.sidebar:
    st.header("🗂️ DocSort AI")
    st.caption("Sorts scanned documents by type and reads their key fields.")
    if health:
        m = health.get("classifier") or {}
        if m.get("test_accuracy"):
            st.metric("Classifier test accuracy", f"{m['test_accuracy']:.1%}")
        st.write("**Types:** " + ", ".join(LABELS[c] for c in health["classes"]))
        if not health["extraction"]:
            st.warning("Field extraction is off (no Gemini key). Documents are still sorted.")
    st.divider()
    st.caption("🔒 **Privacy:** pages are sorted on the server by a local model. Reading fields sends the page "
               "to the free Gemini API, where Google may use it to improve its products — don't upload "
               "personal or confidential documents.")

# ---------------------------------------------------------------- upload
st.title("Sort documents and pull out the important fields")
st.caption("Receipts and invoices → totals, dates and line items. Letters, emails, forms, CVs and papers → "
           "sender, date and a one-line summary. Everything exports to Excel.")

files = st.file_uploader("Drop scans or photos (JPG, PNG, PDF — first page)", type=["jpg", "jpeg", "png", "webp", "pdf"],
                         accept_multiple_files=True)
use_samples = st.button("Try with sample documents") if not files else False
if use_samples:
    st.session_state.samples = True
inputs = [(f.name, f.getvalue(), f.type or "application/octet-stream") for f in files or []]
if not inputs and st.session_state.get("samples"):
    inputs = [(p.name, p.read_bytes(), "image/jpeg") for p in sorted(SAMPLES.glob("*.jpg"))]

if inputs:
    results: list[Result] = []
    progress = st.progress(0.0, text="Sorting…")
    for i, (name, data, mime) in enumerate(inputs, 1):
        cache_key = f"{name}:{len(data)}:{hash(data)}"
        cached = st.session_state.setdefault("cache", {}).get(cache_key)
        if cached is None:
            body, err = api("POST", "/process", files={"file": (name, data, mime)})
            if err:
                st.error(f"{name}: {err}")
                progress.progress(i / len(inputs))
                continue
            cached = st.session_state.cache[cache_key] = body
        results.append(Result.model_validate(cached))
        progress.progress(i / len(inputs), text=f"Sorted {i} of {len(inputs)}")
    progress.empty()

    if results:
        counts = {}
        for r in results:
            counts[r.classification.label] = counts.get(r.classification.label, 0) + 1
        st.subheader(" · ".join(f"{n} {label}" for label, n in counts.items()))
        st.download_button("⬇️ Download Excel report", build_report(results), "docsort-report.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")

        by_name = {name: data for name, data, _ in inputs}
        for r in results:
            c = r.classification
            with st.container(border=True):
                left, right = st.columns([1, 2.4])
                with left:
                    data = by_name.get(r.filename)
                    if data and not data.startswith(b"%PDF"):
                        st.image(data, width="stretch")
                    else:
                        st.caption("📄 PDF — first page")
                with right:
                    st.markdown(f"### {ICONS.get(c.doc_type, '📄')} {c.label}")
                    st.caption(f"{r.filename} · {c.confidence:.0%} sure · {r.latency_ms / 1000:.1f} s")
                    st.progress(c.confidence)
                    if c.unsure:
                        st.warning("Not sure about this one — other candidates: " +
                                   ", ".join(f"{LABELS[t]} {p:.0%}" for t, p in c.top3[1:]))
                    if r.extraction_error:
                        st.info(r.extraction_error)
                    f = r.fields or {}
                    facts = {k.replace("_", " ").capitalize(): v for k, v in f.items()
                             if k not in ("items", "summary") and v not in (None, "", [])}
                    if f.get("summary"):
                        st.write(f["summary"])
                    if facts:
                        st.table({"Field": list(facts), "Value": [show(v) for v in facts.values()]})
                    if f.get("items"):
                        st.dataframe(f["items"], hide_index=True, width="stretch")

#!/usr/bin/env python3
"""
app.py - Streamlit GUI for the Named Entity Recognition system.

Run with:
    streamlit run app.py --server.headless true

Highlights entities (PER, ORG, LOC, MISC, DATE) in the input text, shows a
pandas table of entities, per-category counts, and a CSV download button.
"""

import html
import json
import os

import pandas as pd
import streamlit as st

from ner_engine import extract_entities, ModelNotFoundError, CANONICAL_LABELS

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
METRICS_FILE = "metrics.json"
MODEL_DIR = "./model"

# Entity -> colour (used for the legend and the HTML highlights).
ENTITY_COLOURS = {
    "PER": "#1f77b4",   # blue
    "ORG": "#2ca02c",   # green
    "LOC": "#ff7f0e",   # orange
    "MISC": "#9467bd",  # purple
    "DATE": "#d62728",  # red
}

SAMPLE_TEXTS = {
    "Indian news": (
        "Sundar Pichai met Narendra Modi in Delhi on 5 March 2024 to discuss "
        "Google's expansion in India and the new office in Bengaluru."
    ),
    "Tech": (
        "Apple Inc. is headquartered in Cupertino, California. CEO Tim Cook "
        "announced the new product on 12 January 2023."
    ),
    "International": (
        "The United Nations signed an agreement in Paris on 12 January 2023. "
        "Secretary-General António Guterres attended the meeting."
    ),
}

st.set_page_config(page_title="NER System - Named Entity Recognition", layout="wide")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@st.cache_resource
def load_metrics():
    """Load metrics.json if it exists (cached)."""
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, "r") as f:
            return json.load(f)
    return None


def highlight_text(text, entities):
    """
    Build HTML with entities highlighted using non-overlapping spans.
    Entities are sorted by start position; later entities override earlier ones
    on overlap. Characters are escaped to be safe with unsafe_allow_html.
    """
    if not entities:
        return html.escape(text)

    # Sort by start (descending) so we can splice from the end without
    # disturbing earlier offsets.
    sorted_ents = sorted(entities, key=lambda e: e["start"], reverse=True)
    result = text
    for ent in sorted_ents:
        start, end = ent["start"], ent["end"]
        label = ent["label"]
        colour = ENTITY_COLOURS.get(label, "#666666")
        chunk = html.escape(result[start:end])
        replacement = (
            f"<span style='background-color:{colour}; color:#fff; "
            f"border-radius:4px; padding:1px 6px; font-size:0.85em; "
            f"font-weight:600; margin:1px 0; display:inline-block;' "
            f"title='{html.escape(label)}'>{chunk}"
            f"<span style='font-size:0.65em; opacity:0.9;'>[{html.escape(label)}]</span>"
            f"</span>"
        )
        result = result[:start] + replacement + result[end:]
    return result


def entity_table(entities):
    """Return a pandas DataFrame of entities."""
    rows = [
        {
            "Entity": ent["text"],
            "Label": ent["label"],
            "Confidence": round(ent["score"], 3),
            "Start": ent["start"],
            "End": ent["end"],
        }
        for ent in entities
    ]
    return pd.DataFrame(rows, columns=["Entity", "Label", "Confidence", "Start", "End"])


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Model info")
    metrics = load_metrics()
    if metrics:
        overall = metrics.get("overall", {})
        st.metric("Test F1", f'{overall.get("f1", 0):.4f}')
        st.metric("Test precision", f'{overall.get("precision", 0):.4f}')
        st.metric("Test recall", f'{overall.get("recall", 0):.4f}')
        st.caption(f'Model: {metrics.get("model_name", "unknown")}')
        st.caption(f'Dataset: {metrics.get("dataset", "unknown")}')
        st.caption(f'Epochs: {metrics.get("epochs", "unknown")}')
    else:
        st.info("metrics.json not found. Train with `python train.py`.")
    st.divider()
    st.header("Colour legend")
    for label, colour in ENTITY_COLOURS.items():
        st.markdown(
            f"<span style='background-color:{colour}; color:#fff; "
            f"padding:2px 8px; border-radius:4px; font-size:0.8em;'>{label}</span> "
            f"<b>{label}</b>",
            unsafe_allow_html=True,
        )
    st.divider()
    st.caption("CoNLL-2003 has no DATE label. DATE entities are added with spaCy.")

# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------
st.title("Named Entity Recognition (NER) System")
st.write(
    "Identify **PER** (person), **ORG** (organisation), **LOC** (location), "
    "**MISC** (miscellaneous) and **DATE** entities in text. "
    "The model is fine-tuned on CoNLL-2003; dates are detected with spaCy."
)

if not os.path.isdir(MODEL_DIR):
    st.error(
        "Model not found at ./model. Train it first by running:\n"
        "`python train.py` (or `python train.py --smoke` for a quick test)."
    )

# Sample-text loader using session_state.
if "load_sample" not in st.session_state:
    st.session_state.load_sample = None

col1, col2 = st.columns([5, 1])
with col1:
    text_input = st.text_area(
        "Enter text",
        value=st.session_state.get("text_area", ""),
        height=200,
        placeholder="e.g. Sundar Pichai met Narendra Modi in Delhi on 5 March 2024 ...",
        key="text_area",
    )
with col2:
    st.write("")
    extract_clicked = st.button("Extract Entities", type="primary", use_container_width=True)

sample_cols = st.columns(len(SAMPLE_TEXTS))
for i, (name, sample) in enumerate(SAMPLE_TEXTS.items()):
    with sample_cols[i]:
        if st.button(f"Load sample: {name}", use_container_width=True, key=f"sample_{i}"):
            st.session_state.text_area = sample
            st.rerun()

# Determine the text to process (session_state is updated by the sample buttons).
text_to_process = st.session_state.get("text_area", "")

# Length guard.
if len(text_to_process) > 5000:
    st.warning(
        f"Input is {len(text_to_process)} characters (over the 5000 limit). "
        "It will be truncated for entity extraction."
    )
    text_to_process = text_to_process[:5000]

if extract_clicked:
    if not text_to_process.strip():
        st.warning("Please enter some text before extracting entities.")
    else:
        try:
            entities = extract_entities(text_to_process)
        except ModelNotFoundError as exc:
            st.error(str(exc) + " Run `python train.py` first.")
            entities = []
        except Exception as exc:  # noqa: BLE001 - show a friendly UI error
            st.error(f"Unexpected error during extraction: {exc}")
            entities = []

        # Highlighted text.
        st.subheader("Highlighted text")
        highlighted = highlight_text(text_to_process, entities)
        st.markdown(
            f"<div style='font-size:1.1em; line-height:1.7; "
            f"border:1px solid #ddd; border-radius:6px; padding:1rem; "
            f"background:#fafafa;'>{highlighted}</div>",
            unsafe_allow_html=True,
        )

        # Entity table + counts.
        st.subheader("Entities")
        if entities:
            df = entity_table(entities)
            st.dataframe(df[["Entity", "Label", "Confidence"]], use_container_width=True, hide_index=True)

            counts = df["Label"].value_counts()
            colA, colB = st.columns(2)
            with colA:
                st.write("Count per category")
                st.bar_chart(counts)
            with colB:
                for label in CANONICAL_LABELS:
                    st.metric(label, int(counts.get(label, 0)))

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="entities.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.info("No named entities found.")

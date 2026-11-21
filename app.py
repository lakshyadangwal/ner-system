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

# Entity -> colour (used for the legend, the highlights and the count bar).
# Muted ink tones that stay readable on both light and dark themes.
ENTITY_COLOURS = {
    "PER": "#5b8bd6",   # blue
    "ORG": "#4ea27c",   # green
    "LOC": "#d08a35",   # amber
    "MISC": "#9583c4",  # violet
    "DATE": "#3fa3b0",  # teal
}

ENTITY_NAMES = {
    "PER": "Person",
    "ORG": "Organisation",
    "LOC": "Location",
    "MISC": "Miscellaneous",
    "DATE": "Date",
}

# Colour for the primary button only.
ACCENT = "#b3443d"

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


def tint(hex_colour, alpha):
    """Return an rgba() string for a #rrggbb colour at the given opacity."""
    r, g, b = (int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r}, {g}, {b}, {alpha})"


# Page styles. Colours that must work in both themes use transparency or
# inherit the theme's text colour, so nothing is hard-coded light or dark.
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,500&family=IBM+Plex+Mono:wght@400;500&display=swap');

h1 {{
    font-family: 'Newsreader', Georgia, serif !important;
    font-weight: 500 !important;
    letter-spacing: -0.01em;
}}
h2, h3 {{ font-weight: 500 !important; }}

button[kind="primary"], [data-testid="stBaseButton-primary"] {{
    background-color: {ACCENT} !important;
    border-color: {ACCENT} !important;
    color: #fff !important;
    transition: background-color 150ms cubic-bezier(0.22, 1, 0.36, 1);
}}
button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {{
    background-color: #9c3a34 !important;
    border-color: #9c3a34 !important;
}}

.ner-lede {{ opacity: 0.72; font-size: 1rem; line-height: 1.6; max-width: 44rem; margin: -8px 0 24px; }}
.ner-hint {{ font-size: 0.8rem; opacity: 0.6; margin: 4px 0 4px; }}

.ner-proof {{
    font-family: 'Newsreader', Georgia, serif;
    font-size: 1.2rem;
    line-height: 2;
    padding: 20px 24px;
    border: 1px solid rgba(128, 128, 128, 0.22);
    border-radius: 12px;
}}
.ner-ent {{
    padding: 1px 3px;
    border-radius: 3px 3px 0 0;
    border-bottom: 2px solid;
}}
.ner-tag {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    font-weight: 500;
    margin-left: 4px;
    vertical-align: 0.45em;
}}

.ner-stats {{ font-size: 0.9rem; }}
.ner-stats .row {{ display: flex; justify-content: space-between; padding: 4px 0; }}
.ner-stats .val {{ font-family: 'IBM Plex Mono', monospace; font-variant-numeric: tabular-nums; }}
.ner-stats .big {{ font-size: 2rem; font-family: 'IBM Plex Mono', monospace; font-variant-numeric: tabular-nums; line-height: 1.1; }}
.ner-meta {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; opacity: 0.6; line-height: 1.7; margin-top: 12px; }}

.ner-legend {{ display: flex; flex-direction: column; gap: 8px; font-size: 0.9rem; }}
.ner-legend .item, .ner-counts .item {{ display: flex; align-items: center; gap: 8px; }}
.ner-dot {{ width: 8px; height: 8px; border-radius: 50%; flex: none; }}
.ner-code {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; opacity: 0.6; width: 36px; }}

.ner-bar {{ display: flex; height: 8px; border-radius: 4px; overflow: hidden; gap: 2px; margin: 4px 0 12px; }}
.ner-counts {{ display: flex; flex-wrap: wrap; gap: 8px 24px; font-size: 0.9rem; margin-bottom: 16px; }}
.ner-counts .n {{ font-family: 'IBM Plex Mono', monospace; font-variant-numeric: tabular-nums; opacity: 0.7; }}
</style>
""",
    unsafe_allow_html=True,
)


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
    Entities are sorted by start position; an entity overlapping an earlier
    one is skipped. All text (inside and outside entities) is escaped so it is
    safe with unsafe_allow_html.
    """
    if not entities:
        return html.escape(text)

    sorted_ents = sorted(entities, key=lambda e: (e["start"], e["end"]))
    parts = []
    pos = 0
    for ent in sorted_ents:
        start, end = ent["start"], ent["end"]
        if start < pos:
            continue  # overlaps the previous entity
        label = ent["label"]
        colour = ENTITY_COLOURS.get(label, "#888888")
        name = ENTITY_NAMES.get(label, label)
        parts.append(html.escape(text[pos:start]))
        parts.append(
            f"<span class='ner-ent' style='background-color:{tint(colour, 0.18)}; "
            f"border-color:{colour};' title='{html.escape(name)}'>"
            f"{html.escape(text[start:end])}"
            f"<span class='ner-tag' style='color:{colour};'>{html.escape(label)}</span>"
            f"</span>"
        )
        pos = end
    parts.append(html.escape(text[pos:]))
    return "".join(parts)


def entity_table(entities):
    """
    Return a pandas DataFrame of entities. DATE entities come from spaCy's
    rule-based NER, which has no confidence score, so they show "rule-based".
    """
    rows = [
        {
            "Entity": ent["text"],
            "Label": ent["label"],
            "Confidence": "rule-based" if ent["label"] == "DATE" else round(ent["score"], 3),
            "Start": ent["start"],
            "End": ent["end"],
        }
        for ent in entities
    ]
    return pd.DataFrame(rows, columns=["Entity", "Label", "Confidence", "Start", "End"])


def count_bar(counts):
    """One stacked bar showing the share of each entity type, plus counts."""
    total = int(sum(counts.values()))
    segments = "".join(
        f"<div style='flex:{counts[label]}; background:{ENTITY_COLOURS[label]};'></div>"
        for label in CANONICAL_LABELS
        if counts.get(label, 0)
    )
    items = "".join(
        f"<span class='item'><span class='ner-dot' style='background:{ENTITY_COLOURS[label]};'></span>"
        f"{ENTITY_NAMES[label]} <span class='n'>{int(counts.get(label, 0))}</span></span>"
        for label in CANONICAL_LABELS
    )
    return (
        f"<div class='ner-hint'>{total} {'entity' if total == 1 else 'entities'} found</div>"
        f"<div class='ner-bar'>{segments}</div>"
        f"<div class='ner-counts'>{items}</div>"
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Model")
    metrics = load_metrics()
    if metrics:
        overall = metrics.get("overall", {})
        st.markdown(
            f"""
<div class="ner-stats">
  <div class="ner-hint">Test F1</div>
  <div class="big">{overall.get("f1", 0) * 100:.1f}%</div>
  <div style="height:12px"></div>
  <div class="row"><span>Precision</span><span class="val">{overall.get("precision", 0) * 100:.1f}%</span></div>
  <div class="row"><span>Recall</span><span class="val">{overall.get("recall", 0) * 100:.1f}%</span></div>
</div>
<div class="ner-meta">
  {html.escape(str(metrics.get("model_name", "unknown")))}<br>
  {html.escape(str(metrics.get("dataset", "unknown")))}<br>
  {html.escape(str(metrics.get("epochs", "unknown")))} {"epoch" if metrics.get("epochs") == 1 else "epochs"}
</div>
<div style="height:12px"></div>
""",
            unsafe_allow_html=True,
        )
        if metrics.get("smoke_test"):
            st.warning(
                "This model is from a smoke-test run (small subset, 1 epoch), "
                "so predictions will be poor. Run `python train.py` for full training."
            )
    else:
        st.info("metrics.json not found. Train with `python train.py`.")
    st.divider()
    st.subheader("Entity types")
    legend = "".join(
        f"<div class='item'><span class='ner-dot' style='background:{colour};'></span>"
        f"<span class='ner-code'>{label}</span>{ENTITY_NAMES[label]}</div>"
        for label, colour in ENTITY_COLOURS.items()
    )
    st.markdown(f"<div class='ner-legend'>{legend}</div>", unsafe_allow_html=True)
    st.divider()
    st.caption("CoNLL-2003 has no DATE label, so dates are added with spaCy")

# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------
st.title("NER System")
st.markdown(
    "<div class='ner-lede'>Finds people, organisations, locations and dates in English text. "
    "Names come from DistilBERT fine-tuned on CoNLL-2003 news, dates from spaCy.</div>",
    unsafe_allow_html=True,
)

if not os.path.isdir(MODEL_DIR):
    st.error(
        "Model not found at ./model. Train it first by running:\n"
        "`python train.py` (or `python train.py --smoke` for a quick test)."
    )

# Sample-text loader. The text area is bound to st.session_state["text_area"],
# so the sample buttons update it through an on_click callback (callbacks run
# before the widget is created, which Streamlit requires).
if "text_area" not in st.session_state:
    st.session_state.text_area = ""


def load_sample(sample):
    st.session_state.text_area = sample


col1, col2 = st.columns([5, 1])
with col1:
    text_input = st.text_area(
        "Text",
        height=200,
        placeholder="Sundar Pichai met Narendra Modi in Delhi on 5 March 2024",
        key="text_area",
    )
with col2:
    st.write("")
    extract_clicked = st.button("Extract entities", type="primary", use_container_width=True)

st.markdown("<div class='ner-hint'>Or try a sample</div>", unsafe_allow_html=True)
sample_cols = st.columns(len(SAMPLE_TEXTS))
for i, (name, sample) in enumerate(SAMPLE_TEXTS.items()):
    with sample_cols[i]:
        st.button(
            name,
            use_container_width=True,
            key=f"sample_{i}",
            on_click=load_sample,
            args=(sample,),
        )

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
        st.warning("Enter some text first")
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
        st.markdown(f"<div class='ner-proof'>{highlighted}</div>", unsafe_allow_html=True)

        # Counts + entity table.
        st.subheader("Entities")
        if entities:
            df = entity_table(entities)
            counts = df["Label"].value_counts().to_dict()
            st.markdown(count_bar(counts), unsafe_allow_html=True)
            st.dataframe(df[["Entity", "Label", "Confidence"]], use_container_width=True, hide_index=True)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="entities.csv",
                mime="text/csv",
            )
        else:
            st.info("No named entities found")

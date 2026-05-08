"""Streamlit app body. Imported and re-run by `streamlit run`."""

from __future__ import annotations

# Import pii_mask FIRST so its __init__ sets telemetry-off env vars
# before streamlit / transformers / huggingface_hub get loaded.
from pii_mask import __version__
from pii_mask.inference import load_pii_model, predict_spans
from pii_mask.masking import mask_text, unmask_text
from pii_mask.model_loader import fetch_model

import json
import time
from html import escape
from pathlib import Path

import streamlit as st


_LABEL_COLORS: dict[str, tuple[str, str]] = {
    "KISI_ADI":       ("#a5b4fc", "99, 102, 241"),
    "TCKN":           ("#7dd3fc", "14, 165, 233"),
    "SGK_NO":         ("#7dd3fc", "14, 165, 233"),
    "VKN":            ("#7dd3fc", "14, 165, 233"),
    "IBAN":           ("#67e8f9", "6, 182, 212"),
    "TELEFON":        ("#6ee7b7", "16, 185, 129"),
    "EPOSTA":         ("#6ee7b7", "16, 185, 129"),
    "ADRES":          ("#fcd34d", "245, 158, 11"),
    "TARIH":          ("#cbd5e1", "148, 163, 184"),
    "POLICE_NO":      ("#f9a8d4", "236, 72, 153"),
    "HASAR_DOSYA_NO": ("#f9a8d4", "236, 72, 153"),
    "PLAKA":          ("#fca5a5", "220, 38, 38"),
    "SASI_NO":        ("#fca5a5", "220, 38, 38"),
}


def _render_highlighted(text: str, spans: list[dict]) -> str:
    parts: list[str] = []
    cursor = 0
    for span in sorted(spans, key=lambda s: s["start"]):
        parts.append(escape(text[cursor:span["start"]]))
        fg, rgb = _LABEL_COLORS.get(span["label"], ("#e7e9ee", "120, 120, 120"))
        parts.append(
            f'<span style="background:rgba({rgb},0.18);'
            f'border:1px solid rgba({rgb},0.45);'
            f'border-radius:4px;padding:1px 5px;color:{fg};">'
            f'{escape(text[span["start"]:span["end"]])}'
            f'<span style="font-size:0.7em;opacity:0.7;margin-left:5px;'
            f'letter-spacing:0.05em;font-weight:600;">{span["label"]}</span>'
            f'</span>'
        )
        cursor = span["end"]
    parts.append(escape(text[cursor:]))
    return (
        '<pre style="white-space:pre-wrap;word-wrap:break-word;'
        'font:13px/1.7 ui-monospace,SFMono-Regular,Menlo,monospace;'
        'background:#1a1d24;border:1px solid #2a2e37;border-radius:6px;'
        'padding:18px;margin:0;">' + "".join(parts) + '</pre>'
    )


st.set_page_config(
    page_title="pii-mask-tr",
    page_icon="·",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; max-width: 1200px; }
      h1 { font-weight: 500 !important; letter-spacing: -0.01em; }
      .stTabs [data-baseweb="tab-list"] { gap: 4px; }
      .stTabs [data-baseweb="tab"] { padding: 8px 18px; border-radius: 6px; }
      [data-testid="stMetricValue"] { font-size: 22px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def _bootstrap():
    with st.status("Preparing model …", expanded=False) as status:
        status.update(label="Downloading model from Hugging Face …")
        checkpoint = fetch_model(quiet=True)
        status.update(label="Loading weights …")
        model, tokenizer = load_pii_model(checkpoint)
        status.update(label="Ready.", state="complete")
    return model, tokenizer


@st.cache_resource(show_spinner="Initialising PDF reader …")
def _converter():
    from pii_mask.pdf_ingest import build_converter
    return build_converter()


st.title("Turkish PII detector")
st.caption(
    "Drop a PDF or paste text. Detected entities are highlighted, masked "
    "into round-trippable placeholders, and the mapping stays on this "
    "machine so you can un-mask any LLM response."
)

try:
    model, tokenizer = _bootstrap()
except SystemExit:
    st.error(
        "The model is currently private. Set `HF_TOKEN` "
        "(see https://huggingface.co/settings/tokens) and reload."
    )
    st.stop()
except Exception as exc:  # noqa: BLE001
    st.error(f"Failed to load model: {exc}")
    st.stop()

tab_detect, tab_unmask = st.tabs(["Detect & mask", "Unmask"])

with tab_detect:
    col_in, col_out = st.columns([5, 7], gap="large")

    with col_in:
        st.markdown("##### Input")
        uploaded = st.file_uploader(
            "Upload PDF",
            type=["pdf", "txt"],
            label_visibility="collapsed",
        )
        text_input = st.text_area(
            "or paste text",
            height=320,
            placeholder="Sayın Berkay Gökova, …",
            key="paste_text",
        )
        run = st.button("Detect", type="primary", use_container_width=True)

    with col_out:
        st.markdown("##### Output")

        text: str | None = None
        source_label = ""
        if run:
            if uploaded is not None:
                if uploaded.name.lower().endswith(".pdf"):
                    with st.spinner("Extracting PDF text …"):
                        from pii_mask.pdf_ingest import parse_pdf_bytes
                        text = parse_pdf_bytes(_converter(), uploaded.getvalue(), uploaded.name)
                else:
                    text = uploaded.getvalue().decode("utf-8", errors="replace")
                source_label = uploaded.name
            elif text_input.strip():
                text = text_input
                source_label = "pasted text"
            else:
                st.info("Upload a file or paste text on the left.")

        if text is not None:
            started = time.perf_counter()
            with st.spinner("Detecting entities …"):
                spans = predict_spans(model, tokenizer, text)
            masked, mapping = mask_text(text, spans)
            elapsed = time.perf_counter() - started

            m1, m2, m3 = st.columns(3)
            m1.metric("Entities", len(spans))
            m2.metric("Unique placeholders", len(mapping))
            m3.metric("Time", f"{elapsed:.1f}s")

            st.session_state["last_mapping"] = mapping
            st.session_state["last_masked"] = masked

            view_tab, masked_tab, mapping_tab = st.tabs(
                ["Highlights", "Masked text", "Mapping"]
            )
            with view_tab:
                st.markdown(_render_highlighted(text, spans), unsafe_allow_html=True)
            with masked_tab:
                st.caption(
                    "Copy this and send it to your LLM. Use the Unmask tab "
                    "to restore the response."
                )
                st.code(masked, language=None)
            with mapping_tab:
                st.caption(f"{len(mapping)} unique entities. Stays on this machine.")
                st.json(mapping, expanded=True)

            st.download_button(
                "Download mapping.json",
                data=json.dumps(mapping, ensure_ascii=False, indent=2),
                file_name=f"{Path(source_label).stem or 'document'}.mapping.json",
                mime="application/json",
            )

with tab_unmask:
    st.markdown("##### Reconstruct an LLM response")
    st.caption(
        "Paste the LLM's reply (with `«LABEL_N»` placeholders) and the "
        "mapping from the Detect tab."
    )

    col_a, col_b = st.columns(2, gap="large")
    default_masked = st.session_state.get("last_masked", "")
    default_mapping = json.dumps(
        st.session_state.get("last_mapping", {}),
        ensure_ascii=False,
        indent=2,
    )

    with col_a:
        llm_response = st.text_area(
            "LLM response (with placeholders)",
            value=default_masked,
            height=320,
        )
    with col_b:
        mapping_text = st.text_area(
            "mapping JSON",
            value=default_mapping,
            height=320,
        )

    if st.button("Unmask", type="primary", use_container_width=True):
        try:
            mapping = json.loads(mapping_text or "{}")
        except json.JSONDecodeError as exc:
            st.error(f"mapping is not valid JSON: {exc}")
        else:
            restored = unmask_text(llm_response, mapping)
            st.markdown("##### Restored")
            st.text_area("restored text", restored, height=320, label_visibility="collapsed")

st.caption(f"pii-mask-tr {__version__}")

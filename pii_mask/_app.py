"""Streamlit app body. Imported and re-run by `streamlit run`."""

from __future__ import annotations

# Import pii_mask FIRST so its __init__ sets telemetry-off env vars
# before streamlit / transformers / huggingface_hub get loaded.
from pii_mask import __version__
from pii_mask.inference import load_pii_model, predict_spans
from pii_mask.masking import mask_text, unmask_text
from pii_mask.model_loader import ModelLoadError, fetch_model

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

PAGE_SIZE_CHARS = 4000
PAGE_SNAP_WINDOW = 350
CONTEXT_CHARS_AROUND = 180


def _entity_summary(spans: list[dict]) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for s in spans:
        counts[s["label"]] = counts.get(s["label"], 0) + 1
    return sorted(counts.items(), key=lambda x: (-x[1], x[0]))


def _render_summary_strip(summary: list[tuple[str, int]]) -> str:
    if not summary:
        return ""
    parts = ['<div style="display:flex;flex-wrap:wrap;gap:8px;margin:4px 0 18px 0;">']
    for label, count in summary:
        fg, rgb = _LABEL_COLORS.get(label, ("#e7e9ee", "120, 120, 120"))
        parts.append(
            f'<span style="background:rgba({rgb},0.14);'
            f'border:1px solid rgba({rgb},0.4);border-radius:14px;'
            f'padding:3px 11px;color:{fg};font-size:12px;'
            f'font-weight:500;letter-spacing:0.02em;">'
            f'<strong style="margin-right:6px;font-weight:700;">{count}</strong>{label}'
            "</span>"
        )
    parts.append("</div>")
    return "".join(parts)


def _paginate(text: str) -> list[tuple[int, int]]:
    """Split text into pages of ~PAGE_SIZE_CHARS, snapping breaks to
    paragraph boundaries where possible. Returns list of (start, end)."""
    pages: list[tuple[int, int]] = []
    cursor = 0
    n = len(text)
    while cursor < n:
        end = min(cursor + PAGE_SIZE_CHARS, n)
        if end < n:
            best = end
            for offset in range(PAGE_SNAP_WINDOW):
                fwd = end + offset
                if fwd < n and text[fwd:fwd + 2] == "\n\n":
                    best = fwd + 2
                    break
                back = end - offset
                if back > cursor and text[back:back + 2] == "\n\n":
                    best = back + 2
                    break
            end = best
        pages.append((cursor, end))
        cursor = end
    return pages or [(0, n)]


def _filter_spans_to_page(
    spans: list[dict], page_start: int, page_end: int,
) -> list[dict]:
    out: list[dict] = []
    for s in spans:
        if s["start"] >= page_start and s["end"] <= page_end:
            out.append({
                **s,
                "start": s["start"] - page_start,
                "end": s["end"] - page_start,
            })
    return out


def _extract_contexts(text: str, spans: list[dict]) -> list[dict]:
    """One context window per span (or merged window per cluster).
    Each window is dict(text=..., spans=[...with local offsets]).
    """
    if not spans:
        return []
    spans_sorted = sorted(spans, key=lambda s: s["start"])
    windows: list[list[int]] = []
    for s in spans_sorted:
        ws = max(0, s["start"] - CONTEXT_CHARS_AROUND)
        we = min(len(text), s["end"] + CONTEXT_CHARS_AROUND)
        # snap to whitespace
        while ws > 0 and not text[ws].isspace():
            ws -= 1
            if (s["start"] - ws) > CONTEXT_CHARS_AROUND * 1.3:
                break
        while we < len(text) and not text[we - 1].isspace():
            we += 1
            if (we - s["end"]) > CONTEXT_CHARS_AROUND * 1.3:
                break
        if windows and ws <= windows[-1][1]:
            windows[-1][1] = max(we, windows[-1][1])
        else:
            windows.append([ws, we])
    contexts: list[dict] = []
    for ws, we in windows:
        prefix = "… " if ws > 0 else ""
        suffix = " …" if we < len(text) else ""
        win_text = prefix + text[ws:we] + suffix
        win_spans = [
            {**s,
             "start": s["start"] - ws + len(prefix),
             "end": s["end"] - ws + len(prefix)}
            for s in spans
            if s["start"] >= ws and s["end"] <= we
        ]
        contexts.append({"text": win_text, "spans": win_spans})
    return contexts


def _chip(text_slice: str, label: str) -> str:
    fg, rgb = _LABEL_COLORS.get(label, ("#e7e9ee", "120, 120, 120"))
    return (
        f'<span style="background:rgba({rgb},0.18);'
        f'border:1px solid rgba({rgb},0.45);'
        f'border-radius:4px;padding:1px 6px;color:{fg};white-space:nowrap;">'
        f'{escape(text_slice)}'
        f'<span style="font-size:0.7em;opacity:0.7;margin-left:5px;'
        f'letter-spacing:0.05em;font-weight:600;">{label}</span>'
        "</span>"
    )


def _render_prose(text: str, spans: list[dict]) -> str:
    parts: list[str] = []
    cursor = 0
    for span in sorted(spans, key=lambda s: s["start"]):
        parts.append(escape(text[cursor:span["start"]]))
        parts.append(_chip(text[span["start"]:span["end"]], span["label"]))
        cursor = span["end"]
    parts.append(escape(text[cursor:]))
    body = "".join(parts).replace("\n\n", "</p><p>").replace("\n", "<br>")
    return (
        '<div class="pii-prose">'
        f'<p>{body}</p>'
        "</div>"
    )


def _render_context_cards(contexts: list[dict]) -> str:
    if not contexts:
        return (
            '<div class="pii-prose"><p style="opacity:0.5;text-align:center;">'
            "No entities detected.</p></div>"
        )
    cards: list[str] = []
    for i, ctx in enumerate(contexts, start=1):
        body_parts: list[str] = []
        cursor = 0
        for span in sorted(ctx["spans"], key=lambda s: s["start"]):
            body_parts.append(escape(ctx["text"][cursor:span["start"]]))
            body_parts.append(_chip(ctx["text"][span["start"]:span["end"]], span["label"]))
            cursor = span["end"]
        body_parts.append(escape(ctx["text"][cursor:]))
        body = "".join(body_parts).replace("\n\n", " ").replace("\n", " ")
        cards.append(
            f'<div class="pii-card">'
            f'<div class="pii-card-num">{i}</div>'
            f'<div class="pii-card-body">{body}</div>'
            f'</div>'
        )
    return '<div class="pii-cards">' + "".join(cards) + '</div>'


# ---------------- streamlit body ----------------

st.set_page_config(
    page_title="pii-mask-tr",
    page_icon="·",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
    <style>
      header[data-testid="stHeader"] { display: none; }
      [data-testid="stToolbar"] { display: none; }
      .stDeployButton { display: none; }
      #MainMenu { display: none; }

      .block-container {
        padding-top: 2.2rem;
        padding-bottom: 4rem;
        max-width: 1180px;
      }
      h1 {
        font-weight: 500 !important;
        letter-spacing: -0.015em;
        font-size: 1.6rem !important;
        margin-bottom: 1.2rem !important;
      }
      .stTabs [data-baseweb="tab-list"] { gap: 4px; }
      .stTabs [data-baseweb="tab"] {
        padding: 8px 18px;
        border-radius: 6px;
      }
      [data-testid="stMetricValue"] {
        font-size: 22px;
        font-weight: 500;
      }
      [data-testid="stMetricLabel"] {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        opacity: 0.6;
      }
      /* Prose body — readable typography for the Highlighted view */
      .pii-prose {
        font: 15px/1.75 -apple-system, BlinkMacSystemFont, "Segoe UI",
          system-ui, sans-serif;
        background: #1a1d24;
        border: 1px solid #2a2e37;
        border-radius: 8px;
        padding: 28px 32px;
        color: #e7e9ee;
        max-width: 880px;
        margin: 0 auto;
      }
      .pii-prose p { margin: 0 0 1em 0; }
      .pii-prose p:last-child { margin-bottom: 0; }

      /* Context cards */
      .pii-cards { display: flex; flex-direction: column; gap: 10px; }
      .pii-card {
        display: flex;
        gap: 14px;
        background: #1a1d24;
        border: 1px solid #2a2e37;
        border-radius: 8px;
        padding: 14px 18px;
      }
      .pii-card-num {
        flex: 0 0 auto;
        width: 22px;
        font-size: 11px;
        opacity: 0.4;
        font-variant-numeric: tabular-nums;
        padding-top: 2px;
      }
      .pii-card-body {
        flex: 1;
        font: 14.5px/1.65 -apple-system, BlinkMacSystemFont, "Segoe UI",
          system-ui, sans-serif;
        color: #e7e9ee;
      }

      /* Make Streamlit's textarea paste-comfortable */
      textarea {
        font: 14px/1.6 ui-monospace, SFMono-Regular, Menlo, monospace !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def _bootstrap():
    placeholder = st.empty()
    with placeholder.container():
        with st.status("Preparing model …", expanded=False) as status:
            status.update(
                label="Downloading model from Hugging Face "
                "(~500 MB on first run, then cached) …"
            )
            checkpoint = fetch_model(quiet=True)
            status.update(label="Loading weights …")
            model, tokenizer = load_pii_model(checkpoint)
            status.update(label="Ready", state="complete")
    placeholder.empty()
    return model, tokenizer


@st.cache_resource(show_spinner="Initialising PDF reader …")
def _converter():
    from pii_mask.pdf_ingest import build_converter
    return build_converter()


st.title("Turkish PII detector")

try:
    model, tokenizer = _bootstrap()
except ModelLoadError as exc:
    st.error(exc.hint)
    st.caption(
        f"repo: `{exc.repo}` · revision: `{exc.revision}` · kind: `{exc.kind}`"
    )
    st.stop()
except Exception as exc:  # noqa: BLE001
    st.error(f"Failed to load model: {exc}")
    st.stop()

tab_detect, tab_unmask = st.tabs(["Detect", "Unmask"])

# ============== Detect tab ==============
with tab_detect:
    mode = st.segmented_control(
        "Input mode",
        options=["Text", "PDF"],
        default="Text",
        label_visibility="collapsed",
        key="input_mode",
    )

    text: str | None = None
    source_label = ""

    if mode == "PDF":
        uploaded = st.file_uploader(
            "PDF",
            type=["pdf"],
            label_visibility="collapsed",
        )
        run = st.button("Detect", type="primary", use_container_width=True, key="detect_pdf")
        if run and uploaded is not None:
            with st.spinner("Reading PDF …"):
                from pii_mask.pdf_ingest import parse_pdf_bytes
                text = parse_pdf_bytes(_converter(), uploaded.getvalue(), uploaded.name)
            source_label = uploaded.name
        elif run:
            st.warning("Upload a PDF first.")
    else:
        text_input = st.text_area(
            "Text",
            height=240,
            placeholder=(
                "Sayın Berkay Gökova,\n\n"
                "Hasar dosya numaranız HSR-2026-018472 olarak açılmıştır. "
                "Poliçe no: PL-9831-2025-AKS. TC Kimlik: 19283746501."
            ),
            label_visibility="collapsed",
            key="paste_text",
        )
        run = st.button("Detect", type="primary", use_container_width=True, key="detect_text")
        if run and text_input.strip():
            text = text_input
            source_label = "pasted text"
        elif run:
            st.warning("Paste some text first.")

    if text is not None:
        started = time.perf_counter()
        with st.spinner("Detecting …"):
            spans = predict_spans(model, tokenizer, text)
        masked, mapping = mask_text(text, spans)
        elapsed = time.perf_counter() - started

        st.session_state["last_text"] = text
        st.session_state["last_spans"] = spans
        st.session_state["last_masked"] = masked
        st.session_state["last_mapping"] = mapping
        st.session_state["last_elapsed"] = elapsed
        st.session_state["last_source"] = source_label

    # If we have a result (this run or a prior one within the session), render it.
    if "last_text" in st.session_state:
        result_text: str = st.session_state["last_text"]
        result_spans: list[dict] = st.session_state["last_spans"]
        result_masked: str = st.session_state["last_masked"]
        result_mapping: dict[str, str] = st.session_state["last_mapping"]
        elapsed = st.session_state["last_elapsed"]
        source_label = st.session_state["last_source"]

        st.divider()

        # Top row: metrics on the left, Download on the right.
        m_col1, m_col2, m_spacer, m_dl = st.columns([1, 1, 4, 2])
        m_col1.metric("Entities", len(result_spans))
        m_col2.metric("Time", f"{elapsed:.1f}s")
        with m_dl:
            st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
            st.download_button(
                "Download mapping.json",
                data=json.dumps(result_mapping, ensure_ascii=False, indent=2),
                file_name=f"{Path(source_label).stem or 'document'}.mapping.json",
                mime="application/json",
                use_container_width=True,
            )

        # Summary chip strip
        summary = _entity_summary(result_spans)
        if summary:
            st.markdown(_render_summary_strip(summary), unsafe_allow_html=True)
        else:
            st.caption("No entities detected.")

        view_tab, masked_tab, mapping_tab = st.tabs(
            ["Highlighted", "Masked", "Mapping"]
        )

        with view_tab:
            ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 3, 5])
            with ctrl_col1:
                context_only = st.toggle(
                    "Only entity context",
                    value=False,
                    help=(
                        "Show only the surrounding sentences for each "
                        "detected entity instead of the full document."
                    ),
                    key="context_only_toggle",
                )

            if context_only:
                contexts = _extract_contexts(result_text, result_spans)
                with st.container(height=620, border=False):
                    st.markdown(_render_context_cards(contexts), unsafe_allow_html=True)
            else:
                pages = _paginate(result_text)
                if len(pages) > 1:
                    with ctrl_col2:
                        labels = [
                            f"Page {i+1} of {len(pages)}"
                            for i in range(len(pages))
                        ]
                        page_label = st.selectbox(
                            "Page",
                            options=labels,
                            label_visibility="collapsed",
                            key="page_select",
                        )
                        page_idx = labels.index(page_label)
                else:
                    page_idx = 0
                p_start, p_end = pages[page_idx]
                page_text = result_text[p_start:p_end]
                page_spans = _filter_spans_to_page(result_spans, p_start, p_end)
                with st.container(height=620, border=False):
                    st.markdown(_render_prose(page_text, page_spans), unsafe_allow_html=True)

        with masked_tab:
            with st.container(height=620, border=False):
                st.code(result_masked, language=None)

        with mapping_tab:
            with st.container(height=620, border=False):
                st.json(result_mapping, expanded=True)

# ============== Unmask tab ==============
with tab_unmask:
    col_a, col_b = st.columns(2, gap="large")
    default_masked = st.session_state.get("last_masked", "")
    default_mapping = json.dumps(
        st.session_state.get("last_mapping", {}),
        ensure_ascii=False,
        indent=2,
    )

    with col_a:
        llm_response = st.text_area(
            "Masked text or LLM reply",
            value=default_masked,
            height=320,
        )
    with col_b:
        mapping_text = st.text_area(
            "Mapping",
            value=default_mapping,
            height=320,
        )

    if st.button("Unmask", type="primary", use_container_width=True):
        try:
            mapping = json.loads(mapping_text or "{}")
        except json.JSONDecodeError as exc:
            st.error(f"Mapping is not valid JSON: {exc}")
        else:
            restored = unmask_text(llm_response, mapping)
            with st.container(height=320, border=False):
                st.text_area(
                    "Restored",
                    restored,
                    height=300,
                )

st.markdown(
    f'<div style="text-align:center;opacity:0.4;font-size:11px;'
    f'margin-top:48px;letter-spacing:0.05em;">pii-mask-tr {__version__}</div>',
    unsafe_allow_html=True,
)

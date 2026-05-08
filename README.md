# pii-mask-tr

Detect and mask Turkish personal data in PDFs and text. Uses a fine-tuned BERTurk + CRF model that recognises 13 entity classes — names, TC kimlik, IBAN, plates, addresses, dates, e-mails and more — and rewrites them as round-trippable placeholders so you can safely send a document to an LLM and reconstruct the original from the response.

Everything runs locally. Documents and the placeholder mapping never leave your machine.

## Quickstart

You need [`uv`](https://docs.astral.sh/uv/getting-started/installation/) and a Hugging Face read token (the model is currently private; this step will go away when it's made public).

```sh
# 1. Install uv (skip if you already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Set your HF token
export HF_TOKEN=hf_xxx   # https://huggingface.co/settings/tokens

# 3. Pre-download all models (one-time, ~2 GB)
uvx --from git+https://github.com/berkaygkv/pii-mask-tr pii-mask warm

# 4. Mask a PDF
uvx --from git+https://github.com/berkaygkv/pii-mask-tr pii-mask document.pdf
```

Step 3 is recommended on first install — it downloads the PII model, the Docling PDF reader, and the EasyOCR weights up front with progress feedback. Subsequent runs start instantly. If you skip it, the same downloads happen lazily on first use, but with less visibility.

On Windows (PowerShell):

```powershell
irm https://astral.sh/uv/install.ps1 | iex
$env:HF_TOKEN = "hf_xxx"
uvx --from git+https://github.com/berkaygkv/pii-mask-tr pii-mask warm
uvx --from git+https://github.com/berkaygkv/pii-mask-tr pii-mask document.pdf
```

You'll get three files next to the input:

- `document.masked.txt` — text with `«KISI_ADI_1»`, `«TCKN_1»` … placeholders
- `document.mapping.json` — placeholder → original value
- `document.preview.html` — visual highlights, open in any browser

For a browser UI with detect / mask / unmask in one place:

```sh
uvx --from git+https://github.com/berkaygkv/pii-mask-tr pii-mask-ui
```

The UI has two tabs. **Detect** accepts either a PDF upload or a paste-text input (segmented control), runs detection, and shows: a summary strip with one chip per entity class and its count; a Highlighted view that paginates long documents with an optional "Only entity context" toggle; a Masked view with the placeholder-rewritten output; and a Mapping view with the placeholder→original JSON. **Unmask** accepts an LLM reply plus the mapping and reconstructs the original.

## Round trip

```text
your PDF  ─►  pii-mask  ─►  masked.txt  ─►  ChatGPT/Claude/…  ─►  reply
                                                                   │
your final reply  ◄────────  pii-mask-ui (Unmask tab)  ◄───────────┘
```

The mapping is the only thing you need to restore the original. It stays on disk, in JSON, in plain sight.

## Updating the model

The model is published on Hugging Face as `berkaygkv/pii-model-turkish-v4`, `…-v5`, etc. — one repo per version. The package keeps a list of known revisions ordered newest-first and picks the most recent one available, automatically falling back to older revisions if a newer one hasn't been published yet. **You don't need to set anything to get the latest** — it Just Works as new releases are pushed.

To force a re-check against the Hub (e.g. after a new revision is published while your cache holds an older one):

```sh
uvx --refresh --from git+https://github.com/berkaygkv/pii-mask-tr pii-mask document.pdf
```

To pin a specific revision yourself:

```sh
pii-mask --model-revision v4 document.pdf
```

## Entity classes

`KISI_ADI` `TCKN` `SGK_NO` `VKN` `TELEFON` `EPOSTA` `ADRES` `TARIH` `POLICE_NO` `HASAR_DOSYA_NO` `PLAKA` `SASI_NO` `IBAN`

## Performance

The first run downloads the model from Hugging Face (~500 MB) and caches it under `~/.cache/pii-mask-tr/`. We use HF's Rust-based parallel downloader (`hf_transfer`) by default, which makes the first-run download 5-10× faster than the Python implementation. Set `HF_HUB_ENABLE_HF_TRANSFER=0` to opt out.

Subsequent runs are local — model load is ~5 s, inference is ~1-3 s per page.

## Privacy

This is a privacy-first tool. Everything that matters runs on your machine.

**What never leaves your machine:** the document text, the detected spans, the placeholder mapping, anything you paste into the UI, anything you upload. There is no analytics, no usage reporting, no error reporting. The Streamlit UI binds to `localhost` only.

**What does go out, and only when:**

- **First-run model download.** The BERTurk + CRF checkpoint is fetched from `huggingface.co/berkaygkv/pii-model-turkish-v4` once and cached at `~/.cache/pii-mask-tr/`. Your `HF_TOKEN` is sent as the `Authorization` header. Subsequent runs are local-only.
- **First-run PDF dependencies.** The first time you process a PDF, [Docling](https://github.com/DS4SD/docling) downloads its layout / table-structure models from Hugging Face, and EasyOCR downloads its recognition weights from the JaidedAI GitHub releases. Both are one-time, model-only downloads. No telemetry.

**Telemetry disabled by default.** The package sets these on import, before any framework loads:

- `HF_HUB_DISABLE_TELEMETRY=1` — Hugging Face Hub + Transformers usage reporting
- `STREAMLIT_BROWSER_GATHER_USAGE_STATS=false` — Streamlit analytics
- `DO_NOT_TRACK=1` — community anti-telemetry convention

You can override any of these by setting the env var to the opposite value before running.

**Paranoid mode.** Once the model and PDF deps are cached, you can refuse all network access:

```sh
pii-mask --offline document.pdf
```

This sets `HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1`. Any attempt to fetch anything will fail loudly instead of silently dialing out.

## License

MIT.

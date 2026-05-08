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

# 3. Mask a PDF
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

## Round trip

```text
your PDF  ─►  pii-mask  ─►  masked.txt  ─►  ChatGPT/Claude/…  ─►  reply
                                                                   │
your final reply  ◄────────  pii-mask-ui (Unmask tab)  ◄───────────┘
```

The mapping is the only thing you need to restore the original. It stays on disk, in JSON, in plain sight.

## Updating the model

The model lives at `berkaygkv/pii-mask-turkish` on Hugging Face, with one revision per release (`v5`, `v6`, …). Each `pii-mask-tr` release pins a default. To get a newer one:

```sh
uvx --refresh --from git+https://github.com/berkaygkv/pii-mask-tr pii-mask document.pdf
```

To pin a specific revision yourself:

```sh
pii-mask --model-revision v6 document.pdf
```

## Entity classes

`KISI_ADI` `TCKN` `SGK_NO` `VKN` `TELEFON` `EPOSTA` `ADRES` `TARIH` `POLICE_NO` `HASAR_DOSYA_NO` `PLAKA` `SASI_NO` `IBAN`

## Privacy

This is a privacy-first tool. Everything that matters runs on your machine.

**What never leaves your machine:** the document text, the detected spans, the placeholder mapping, anything you paste into the UI, anything you upload. There is no analytics, no usage reporting, no error reporting. The Streamlit UI binds to `localhost` only.

**What does go out, and only when:**

- **First-run model download.** The BERTurk + CRF checkpoint is fetched from `huggingface.co/berkaygkv/pii-mask-turkish` once and cached at `~/.cache/pii-mask-tr/`. Your `HF_TOKEN` is sent as the `Authorization` header. Subsequent runs are local-only.
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

# 🎙️ Discussion Stream Analyzer

Upload up to 30 discussion transcripts (podcasts, meetings, interviews) and get, **for each document separately**:

- an overview paragraph followed by **Key Topics Discussed** (bold title: description)
- a **topic timeline** showing which topic was discussed in which time window
- key terms and share of the discussion per topic
- a safety scan for sensitive keywords (password, confidential, ...)
- downloads: Markdown report, CSV, JSON, and a ZIP of all reports

It pairs with the speech-to-text app: the `.txt` transcripts that app writes (`[00:00:01.00 - 00:00:03.00] speaker_0: ...`) load directly.

<!-- Add a screenshot: save it as docs/screenshot.png and uncomment the next line -->
<!-- ![App screenshot](docs/screenshot.png) -->

## How it works

1. **Parse** each file into turns (timestamps and speaker names are optional; long paragraphs are split at sentence boundaries).
2. **Embed** every turn once with a sentence-embedding model (MiniLM or BGE).
3. **Segment**: compare the text before and after every gap and cut where the subject changes (TextTiling-style).
4. **Cluster** segments into topics (average-linkage on cosine distance, number of topics chosen by silhouette score, tiny topics merged).
5. **Label** topics with c-TF-IDF key terms (the idea behind BERTopic).
6. **Summarise** in one batched pass: an instruction-tuned LLM (Qwen2.5) writes every topic title, description and overview for all documents. Without a GPU, choose *No AI model* and key sentences are used instead.

The analysis runs in a background thread, so reloading the browser tab never loses a run.

## Quick start (local)

```bash
git clone https://github.com/<your-username>/discussion-stream-analyzer.git
cd discussion-stream-analyzer
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
# install PyTorch for your machine first: https://pytorch.org/get-started/locally/
pip install -r requirements.txt
streamlit run app.py
```

The first run downloads the models (about 100 MB for the embedder, 3 to 6 GB for Qwen). The AI summary writer needs a CUDA GPU.

## Quick start (Google Colab)

Open `notebooks/colab_launcher.ipynb` in Colab, set **Runtime > Change runtime type > T4 GPU**, set `REPO_URL` in the first cell, and run all cells. Open the printed `trycloudflare.com` link and keep the last cell running.
The link is public, so don't share it.

## Transcript formats

All of these work (speakers and timestamps are optional):

```
[00:01:23] Alice: text
Alice: text
00:01:23 text
plain lines of text
```

## Project layout

```
discussion-stream-analyzer/
├── app.py                     # Streamlit entry point
├── stream_analyzer/
│   ├── config.py              # limits, model choices, colours
│   ├── parsing.py             # raw text -> turns
│   ├── text.py                # stop words, key sentences, c-TF-IDF keywords
│   ├── embeddings.py          # sentence embeddings (+ offline TF-IDF fallback)
│   ├── segmentation.py        # topic boundaries and clustering
│   ├── pipeline.py            # analyse one document (no AI model needed)
│   ├── llm.py                 # Qwen prompts and batched generation
│   ├── job.py                 # background job: all documents, progress state
│   ├── exporters.py           # Markdown reports, saved runs
│   └── ui.py                  # Streamlit page
├── tests/                     # pytest suite (runs offline, no GPU)
├── notebooks/colab_launcher.ipynb
├── .streamlit/config.toml     # theme and server settings
├── requirements.txt
└── requirements-dev.txt
```

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

The tests use the offline TF-IDF embedder, so they need neither a GPU nor a model download.

## Settings (sidebar)

| Setting | Effect |
|---|---|
| Topic detection model | *Fast* MiniLM-L6, or *Better* BGE-small |
| Summary writer | Qwen2.5-1.5B (fast), Qwen2.5-3B (better), or no AI model (instant) |
| AI summary for every timeline block | Off by default; turning it on is slower |
| Most topics per document | Upper limit for the clustering step |
| Smallest segment | Smaller = finer timeline |
| Topic-change sensitivity | Higher = more topic changes detected |

Every run is saved in `outputs/<timestamp>/` (reports and `results.json`); use **Load last saved run** to reopen it.

## Troubleshooting

- **"No GPU found"**: pick *No AI model* in the sidebar, or switch to a GPU runtime.
- **Topics too coarse or too fine**: adjust *Topic-change sensitivity* and *Smallest segment*, or try the BGE model.
- **Page doesn't load in Colab**: run `!tail -n 40 streamlit_logs.txt`.

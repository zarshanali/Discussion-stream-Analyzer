"""The Streamlit page: sidebar, upload panel, live progress and results."""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

from .config import EMBEDDERS, LLMS, MAX_DOCS, PALETTE
from .embeddings import Embedder
from .exporters import load_last_run, safe, to_markdown
from .job import BUSY, COLOR, LABEL, WEIGHT, Job
from .llm import LLM
from .parsing import decode_bytes

CSS = """<style>
.block-container{padding-top:1.3rem;max-width:1650px}
header[data-testid="stHeader"]{background:transparent}
.hero{display:flex;justify-content:space-between;align-items:center;gap:1rem;flex-wrap:wrap;padding:1.35rem 1.8rem;border-radius:1.1rem;color:#fff;background:linear-gradient(120deg,#312e81 0%,#4f46e5 48%,#0ea5e9 100%);box-shadow:0 10px 28px rgba(79,70,229,.25);margin-bottom:1.1rem}
.hero-t{font-size:1.85rem;font-weight:800;letter-spacing:-.02em;line-height:1.2}.hero-s{opacity:.92;margin-top:.25rem;font-size:.95rem}
.chips{display:flex;gap:.5rem;flex-wrap:wrap}.chip{background:rgba(255,255,255,.16);border:1px solid rgba(255,255,255,.3);padding:.3rem .85rem;border-radius:2rem;font-size:.8rem;font-weight:600}
.kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.75rem;margin:.35rem 0 .8rem}
.kpi{background:#fff;border:1px solid #e5e7eb;border-top:3px solid var(--accent,#4f46e5);border-radius:.8rem;padding:.65rem .95rem;box-shadow:0 1px 3px rgba(15,23,42,.06)}
.kpi-l{font-size:.78rem;color:#64748b;font-weight:700}.kpi-v{font-size:1.5rem;font-weight:800;color:#0f172a;line-height:1.3}.kpi-s{font-size:.78rem;color:#64748b}
.sec{font-size:.9rem;color:#475569;font-weight:700;margin:1rem 0 .4rem}
.tiles{display:flex;flex-wrap:wrap;gap:.35rem}.tile{width:2.15rem;height:2.15rem;border-radius:.55rem;color:#fff;font-size:.74rem;font-weight:700;display:flex;align-items:center;justify-content:center}
.tile.busy{animation:pulse 1.2s ease-in-out infinite}@keyframes pulse{50%{opacity:.55}}
.legend{display:flex;flex-wrap:wrap;gap:.9rem;margin-top:.45rem;font-size:.75rem;color:#64748b}.legend i{display:inline-block;width:.7rem;height:.7rem;border-radius:.2rem;margin-right:.3rem}
.tl{display:flex;height:1.9rem;border-radius:.55rem;overflow:hidden;background:#eef2f7}.tl span{min-width:3px}
.tcard{padding:.7rem .95rem;margin:.45rem 0;border-radius:.7rem;background:#f8fafc;border:1px solid #eef2f7;border-left:5px solid var(--c,#4f46e5);line-height:1.6;color:#0f172a}
.tcard b{font-variant-numeric:tabular-nums}.tag{display:inline-block;padding:.05rem .65rem;border-radius:1rem;color:#fff;font-size:.76rem;font-weight:700;margin-left:.5rem;background:var(--c,#4f46e5)}
.kw{color:#64748b;font-size:.8rem;margin-top:.2rem}
@media (max-width:1100px){.kpis{grid-template-columns:repeat(2,minmax(0,1fr))}}
</style>"""


@st.cache_resource(show_spinner=False)
def registry() -> dict:
    return {"job": None, "loaded": None}              # shared by every browser tab, survives page reloads


@st.cache_resource(show_spinner=False)
def get_embedder(name: str) -> Embedder:
    return Embedder(name)


@st.cache_resource(show_spinner=False)
def get_llm(name: str) -> LLM:                       # an exception is NOT cached, so fixing the runtime and retrying works
    return LLM(name)


def fmt_dur(sec: float) -> str:
    sec = int(round(max(sec or 0.0, 0.0)))
    h, rem = divmod(sec, 3600)
    return f"{h}:{rem // 60:02d}:{rem % 60:02d}" if h else f"{rem // 60}:{rem % 60:02d}"


def show_table(rows, **kw) -> None:
    try:
        st.dataframe(rows, width="stretch", hide_index=True, **kw)
    except Exception:
        st.dataframe(rows, use_container_width=True, hide_index=True, **kw)


def kpi(label: str, value: str, sub: str = "", accent: str = "#4f46e5") -> str:
    return (f'<div class="kpi" style="--accent:{accent}"><div class="kpi-l">{html.escape(label)}</div>'
            f'<div class="kpi-v">{html.escape(value)}</div><div class="kpi-s">{html.escape(sub)}</div></div>')


def kpi_row(*cards: str) -> None:
    st.markdown('<div class="kpis">' + "".join(cards) + "</div>", unsafe_allow_html=True)


def tiles_html(files: List[dict]) -> str:
    tiles = "".join(f'<div class="tile{" busy" if f["status"] in BUSY else ""}" style="background:{COLOR[f["status"]]}" '
                    f'title="{html.escape(f["name"])} - {LABEL[f["status"]][2:]}">{i}</div>' for i, f in enumerate(files, 1))
    legend = "".join(f'<span><i style="background:{COLOR[s]}"></i>{LABEL[s].split(" ", 1)[1]}</span>' for s in COLOR)
    return f'<div class="tiles">{tiles}</div><div class="legend">{legend}</div>'


def live_panel(watching: bool) -> None:
    job: Optional[Job] = registry()["job"]
    if job is None:
        st.info("Progress for every document appears here when you press **Analyze**.")
        return
    files = job.snapshot()
    n_done, n_fail = sum(f["status"] == "done" for f in files), sum(f["status"] == "failed" for f in files)
    words = sum(f["words"] for f in files if f["status"] == "done")
    st.progress(min(max(job.frac, 0.0), 1.0), text=f"{job.phase}  ·  {n_done + n_fail}/{len(files)} documents  ·  {fmt_dur(job.elapsed)} elapsed")
    if job.error:
        st.error(f"The run stopped unexpectedly: {job.error}")
    kpi_row(kpi("Documents finished", f"{n_done + n_fail} / {len(files)}", f"{n_done} done · {n_fail} failed"),
            kpi("Elapsed time", fmt_dur(job.elapsed), "running" if not job.done else "finished", "#0ea5e9"),
            kpi("Words analysed", f"{words:,}", f"{sum(f['words'] for f in files):,} in total", "#8b5cf6"),
            kpi("Speed", f"{words / job.elapsed:,.0f} words/s" if job.elapsed > 0 else "-", "finished documents only", "#10b981"))
    st.markdown('<div class="sec">Documents</div>' + tiles_html(files), unsafe_allow_html=True)
    cc = st.column_config
    show_table([{"#": i, "Document": f["name"], "Status": LABEL[f["status"]], "Progress": round(100 * WEIGHT[f["status"]]),
                 "Words": f["words"], "Segments": f["segments"] or None, "Topics": f["topics"] or None,
                 "Time (s, est.)": round(f["sec"], 1), "Note": f["error"]} for i, f in enumerate(files, 1)],
               column_config={"Progress": cc.ProgressColumn("Progress", min_value=0, max_value=100, format="%d%%")})
    if job.timing:
        with st.expander("Where the time went"):
            show_table([{"Step": k, "Seconds": round(v, 2)} for k, v in job.timing.items()])
    if watching and job.done:
        st.rerun()


def topic_colors(r: dict) -> Dict[int, str]:
    return {t["topic_id"]: PALETTE[i % len(PALETTE)] for i, t in enumerate(r["topics"])}


def timeline_html(r: dict) -> str:
    col = topic_colors(r)
    bar = "".join(f'<span style="flex:{b["words"]};background:{col[b["topic_id"]]}" title="{html.escape(b["topic"])} · '
                  f'{html.escape(b["start"])} → {html.escape(b["end"])}"></span>' for b in r["timeline"])
    legend = "".join(f'<span><i style="background:{col[t["topic_id"]]}"></i>{html.escape(t["topic"])} · {t["share_pct"]:.0f}%</span>'
                     for t in r["topics"])
    cards = "".join(
        f'<div class="tcard" style="--c:{col[b["topic_id"]]}"><b>{html.escape(b["start"])} → {html.escape(b["end"])}</b>'
        f'<span class="tag">{html.escape(b["topic"])}</span><div>{html.escape(b["summary"])}</div>'
        + (f'<div class="kw">Key terms: {html.escape(", ".join(b["keywords"]))}</div>' if b["keywords"] else "") + "</div>"
        for b in r["timeline"])
    return f'<div class="tl">{bar}</div><div class="legend">{legend}</div>{cards}'


def render_doc(r: dict, key: str) -> None:
    kpi_row(kpi("Words", f"{r['words']:,}", f"{r['n_turns']} turns", "#8b5cf6"), kpi("Topics", str(len(r["topics"])), f"{len(r['segments'])} segments", "#e11d48"),
            kpi("Speakers", str(len(r["speakers"])) or "n/a", ", ".join(r["speakers"][:3]) or "none detected", "#0ea5e9"),
            kpi("Timestamps", "found" if r["has_timestamps"] else "none", "positions are turn numbers" if not r["has_timestamps"] else "ranges use them", "#10b981"))
    st.markdown('<div class="sec">Overview</div>', unsafe_allow_html=True)
    st.markdown(r["overview"])
    st.markdown('<div class="sec">Topic timeline</div>', unsafe_allow_html=True)
    st.markdown(timeline_html(r), unsafe_allow_html=True)
    st.markdown('<div class="sec">Topics</div>', unsafe_allow_html=True)
    show_table([{"Topic": t["title"], "Share (%)": t["share_pct"], "Segments": t["segments"], "Key terms": ", ".join(t["keywords"]), "When": t["when"]}
                for t in r["topics"]], column_config={"Share (%)": st.column_config.ProgressColumn("Share of discussion", min_value=0, max_value=100, format="%.0f%%")})
    flagged = [s for s in r["segments"] if s["safety"] != "PASS"]
    st.markdown('<div class="sec">Safety scan</div>', unsafe_allow_html=True)
    if flagged:
        st.warning(f"{len(flagged)} segment(s) contain sensitive keywords.")
        show_table([{"Segment": s["segment"], "When": f"{s['start']} → {s['end']}", "Flag": s["safety"]} for s in flagged])
    else:
        st.success("No sensitive keywords detected.")
    with st.expander("Segments and raw text"):
        for s in r["segments"]:
            st.markdown(f"**Segment {s['segment']}** · {s['start']} → {s['end']} · `{s['topic']}`")
            st.write(s["summary"])
            st.code(s["text"], language="text")
    seg = pd.DataFrame([{k: (", ".join(v) if k == "keywords" else v) for k, v in s.items() if k != "text"} for s in r["segments"]])
    c1, c2, c3 = st.columns(3)
    c1.download_button("⬇ Report (.md)", to_markdown(r), f"{safe(r['name'])}.md", "text/markdown", key=f"md_{key}")
    c2.download_button("⬇ Segments (.csv)", seg.to_csv(index=False), f"{safe(r['name'])}_segments.csv", "text/csv", key=f"csv_{key}")
    c3.download_button("⬇ Full result (.json)", json.dumps(r, indent=2), f"{safe(r['name'])}.json", "application/json", key=f"js_{key}")


def render_results() -> None:
    reg = registry()
    job: Optional[Job] = reg["job"]
    if job is not None and job.done and job.results:
        results, meta, zipb = job.results, dict(embedder=job.embedder.name, llm=job.llm.name if job.llm else "extractive"), job.zip
    elif job is None and reg["loaded"]:
        results, meta, zipb = reg["loaded"]["results"], reg["loaded"]["meta"], b""
    else:
        st.info("Results appear here when the run finishes. Each document gets its own overview, topic timeline and downloads.")
        return
    st.caption(f"Embeddings: {meta.get('embedder')} · Summaries: {meta.get('llm')}")
    pick = st.selectbox("Document", ["All documents"] + list(results))
    if pick == "All documents":
        show_table([{"Document": n, "Words": r["words"], "Topics": len(r["topics"]),
                     "Main topics": " | ".join(t["title"] for t in sorted(r["topics"], key=lambda t: -t["share_pct"])[:3]),
                     "Overview": r["overview"].split("\n\n")[0]} for n, r in results.items()])
        if zipb:
            st.download_button(f"📦 All {len(results)} reports (.zip)", zipb, "reports.zip", "application/zip")
    else:
        render_doc(results[pick], key=safe(pick))


def sidebar() -> dict:
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        emb = st.selectbox("Topic detection model", list(EMBEDDERS), help="Reads the meaning of every turn. 'Fast' is about twice as quick.")
        llm = st.selectbox("Summary writer", list(LLMS), index=0, help="Writes topic titles and the overview. Needs a GPU. "
                           "'No AI model' uses key sentences and finishes instantly.")
        blocks = st.toggle("AI summary for every timeline block", value=False, disabled=LLMS[llm] is None,
                           help="Slower. Off = timeline blocks use the most central sentences.")
        st.divider()
        max_topics = st.slider("Most topics per document", 2, 12, 6)
        min_turns = st.slider("Smallest segment (turns)", 3, 20, 5, help="Smaller = finer timeline.")
        sens = st.slider("Topic-change sensitivity", 1, 9, 6, help="Higher = more topic changes detected.")
        st.divider()
        if st.button("Load last saved run"):
            data = load_last_run()
            if data:
                registry().update(job=None, loaded=data)
                st.rerun()
            st.warning("No saved run found.")
        try:
            import torch
            gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU mode)"
        except Exception:
            gpu = "unknown"
        st.caption(f"**GPU:** {gpu}")
    return dict(embedder=EMBEDDERS[emb], llm=LLMS[llm], ai_blocks=blocks, max_topics=max_topics, min_turns=min_turns,
                max_turns=max(40, 3 * min_turns), sens=sens, window=60)


def start_job(files, cfg: dict) -> None:
    docs: Dict[str, str] = {}
    for f in files:
        base, name, i = Path(f.name).stem, Path(f.name).stem, 2
        while name in docs:
            name, i = f"{base} ({i})", i + 1                 # same file name twice: keep both, never overwrite
        docs[name] = decode_bytes(f.getvalue())
    llm = None
    with st.status("Loading models ... (the first run downloads them)", expanded=True) as status:
        emb = get_embedder(cfg["embedder"])
        if emb.model is None:
            st.warning(f"Sentence model unavailable ({emb.error[:120]}); using the TF-IDF fallback - topics will be less accurate.")
        if cfg["llm"]:
            try:
                llm = get_llm(cfg["llm"])
            except Exception as e:
                st.warning(f"Could not load {cfg['llm']}: {e}. Using key sentences instead.")
        status.update(label="Models ready", state="complete", expanded=False)
    registry().update(job=Job(docs, cfg, emb, llm), loaded=None)
    st.session_state.uploader_key += 1                       # empties the uploader so the files stop using RAM
    st.rerun()


def upload_panel(running: bool, cfg: dict) -> None:
    ss = st.session_state
    st.subheader("1 · Upload")
    ss.setdefault("uploader_key", 0)
    files = st.file_uploader(f"Drop up to {MAX_DOCS} transcripts (.txt, .md, .log)", type=["txt", "md", "log"],
                             accept_multiple_files=True, disabled=running, key=f"up_{ss.uploader_key}") or []
    too_many = len(files) > MAX_DOCS
    if too_many:
        st.error(f"You selected {len(files)} files - the limit is {MAX_DOCS}.")
    go = st.button("🚀 Analyze", type="primary", disabled=not files or too_many or running)
    if running:
        job = registry()["job"]
        if st.button("⏹ Stop after the current step"):
            job.cancel.set()
        st.caption("Stopping..." if job.cancel.is_set() else "Running in the background - you can switch tabs or reload the page.")
    if go:
        start_job(files, cfg)
    job = registry()["job"]
    if job is not None and job.done and job.out_dir:
        st.success(f"Finished {len(job.results)} document(s) in {fmt_dur(job.elapsed)}. Reports saved in {job.out_dir}")


def main() -> None:
    st.set_page_config(page_title="Discussion Stream Analyzer", page_icon="🎙️", layout="wide", initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)
    cfg = sidebar()
    st.markdown('<div class="hero"><div><div class="hero-t">🎙️ Discussion Stream Analyzer</div><div class="hero-s">Topics, a timeline and a summary '
                f'for up to {MAX_DOCS} transcripts at once - each document gets its own result</div></div><div class="chips">'
                '<span class="chip">Topic timeline</span><span class="chip">Key topics overview</span><span class="chip">Safety scan</span></div></div>',
                unsafe_allow_html=True)
    reg = registry()
    left, right = st.columns([1, 2.3], gap="large")
    with left:
        upload_panel(reg["job"] is not None and not reg["job"].done, cfg)
    job = reg["job"]
    running = job is not None and not job.done
    with right:
        tab_live, tab_res = st.tabs(["📊 Live progress", "📄 Results"])
        with tab_live:
            st.fragment(run_every=1.0 if running else None)(live_panel)(running)
        with tab_res:
            render_results()


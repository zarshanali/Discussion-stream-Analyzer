"""Runs a whole analysis in a BACKGROUND THREAD; the page only reads the job's state (so a reload never loses a run)."""
from __future__ import annotations

import io
import json
import threading
import time
import zipfile
from typing import Dict, List, Optional

from .config import OUT
from .embeddings import Embedder
from .exporters import safe, to_markdown
from .llm import BLOCK_P, INTRO_P, LLM, TOPIC_P, overview_md, title_desc
from .parsing import parse
from .pipeline import analyze_one
from .text import dialogue, top_sents

WEIGHT = {"queued": 0.0, "parsing": 0.05, "embedding": 0.2, "analysing": 0.5, "summarising": 0.75, "done": 1.0, "failed": 1.0}
LABEL = {"queued": "⏳ Queued", "parsing": "📖 Reading", "embedding": "🧬 Embedding", "analysing": "🧭 Finding topics",
         "summarising": "🧠 Summarising", "done": "✅ Done", "failed": "❌ Failed"}
COLOR = {"queued": "#cbd5e1", "parsing": "#38bdf8", "embedding": "#818cf8", "analysing": "#6366f1", "summarising": "#a855f7",
         "done": "#10b981", "failed": "#ef4444"}
BUSY = ("parsing", "embedding", "analysing", "summarising")


class Job:
    def __init__(self, docs: Dict[str, str], cfg: dict, embedder: Embedder, llm: Optional[LLM]):
        self.docs, self.cfg, self.embedder, self.llm = docs, cfg, embedder, llm
        self.files = {n: dict(name=n, status="queued", turns=0, words=0, segments=0, topics=0, sec=0.0, error="") for n in docs}
        self.results: Dict[str, dict] = {}
        self.timing: Dict[str, float] = {}
        self.phase, self.frac, self.error = "Starting ...", 0.0, ""
        self.t0, self.t1, self.out_dir, self.zip = time.time(), None, "", b""
        self.cancel, self.lock = threading.Event(), threading.RLock()
        threading.Thread(target=self._main, daemon=True, name="topics-job").start()

    done = property(lambda self: self.t1 is not None)
    elapsed = property(lambda self: (self.t1 or time.time()) - self.t0)

    def upd(self, name: str, **kw) -> None:
        with self.lock:
            self.files[name].update(kw)

    def say(self, text: str, frac: float) -> None:
        self.phase, self.frac = text, max(self.frac, min(frac, 0.99))

    def snapshot(self) -> List[dict]:
        with self.lock:
            return [dict(f) for f in self.files.values()]

    def _main(self) -> None:
        try:
            self._run()
            if self.results:
                self._save()
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
            for n, f in self.files.items():
                if f["status"] not in ("done", "failed"):
                    self.upd(n, status="failed", error=self.error)
        finally:
            self.t1, self.phase, self.frac = time.time(), "Done", 1.0

    def _run(self) -> None:
        cfg, t = self.cfg, time.time()
        self.say("Reading transcripts ...", 0.02)
        parsed: Dict[str, List[dict]] = {}
        for n, raw in self.docs.items():
            self.upd(n, status="parsing")
            turns = parse(raw)
            if not turns:
                self.upd(n, status="failed", error="No readable text in this file")
                continue
            parsed[n] = turns
            self.upd(n, status="embedding", turns=len(turns), words=sum(len(x["text"].split()) for x in turns))
        self.timing["read"] = time.time() - t
        if not parsed:
            return
        t = time.time()
        self.say(f"Embedding {sum(len(v) for v in parsed.values())} turns ({self.embedder.name}) ...", 0.1)
        E = self.embedder.encode([x["text"] for n in parsed for x in parsed[n]])
        self.timing["embed"] = time.time() - t
        t, a, analysed = time.time(), 0, {}
        for k, (n, turns) in enumerate(parsed.items()):
            if self.cancel.is_set():
                break
            self.upd(n, status="analysing")
            self.say(f"Finding topics in {n} ({k + 1}/{len(parsed)}) ...", 0.25 + 0.3 * k / len(parsed))
            t1 = time.time()
            analysed[n] = analyze_one(turns, E[a:a + len(turns)], cfg)
            a += len(turns)
            res = analysed[n][0]
            self.upd(n, status="summarising", segments=len(res["segments"]), topics=len(res["topics"]), sec=time.time() - t1)
        self.timing["topics"] = time.time() - t
        if self.cancel.is_set():
            for n, f in self.files.items():
                if f["status"] not in ("done", "failed"):
                    self.upd(n, status="failed", error="stopped by the user")
            return
        t = time.time()
        self._summarise(parsed, analysed)
        self.timing["summaries"] = time.time() - t
        tot = sum(self.timing.values())
        words = sum(f["words"] for n, f in self.files.items() if n in analysed) or 1
        for n in analysed:
            f = self.files[n]
            share = f["words"] / words
            self.upd(n, status="done", sec=f["sec"] + (self.timing["embed"] + self.timing["summaries"]) * share)
        self.timing["total"] = tot

    def _summarise(self, parsed, analysed) -> None:
        """Every topic title/description, every overview (and optional block summary) of ALL documents in one batched pass."""
        cfg, jobs = self.cfg, []
        for n, (res, ex) in analysed.items():
            if self.llm:
                for tp in res["topics"]:
                    jobs.append((n, "topic", tp["topic_id"], TOPIC_P.format(ex["topics"][tp["topic_id"]])))
                kws = ", ".join(dict.fromkeys(k for tp in res["topics"] for k in tp["keywords"][:2]))
                jobs.append((n, "intro", 0, INTRO_P.format(dialogue(parsed[n][:12])[:2200], kws)))
                if cfg["ai_blocks"]:
                    jobs += [(n, "block", i, BLOCK_P.format(x)) for i, x in enumerate(ex["blocks"])]
        outs = []
        if jobs:
            self.say(f"Writing {len(jobs)} summaries with {self.llm.name.split('/')[-1]} ...", 0.55)
            outs = self.llm.generate([j[3] for j in jobs], progress=lambda f: self.say("Writing summaries ...", 0.55 + 0.4 * f),
                                     stop=self.cancel.is_set)
        got = {(n, kind, i): o for (n, kind, i, _), o in zip(jobs, outs)}
        for n, (res, ex) in analysed.items():
            for tp in res["topics"]:
                title, desc = title_desc(got.get((n, "topic", tp["topic_id"]), ""))
                tp["title"] = title or tp["title"]
                tp["description"] = desc or " ".join(top_sents(ex["topics"][tp["topic_id"]], 2))
                tp["topic"] = tp["title"]
            names = {tp["topic_id"]: tp["title"] for tp in res["topics"]}
            for s in res["segments"]:
                s["topic"] = names[s["topic_id"]]
            for i, b in enumerate(res["timeline"]):
                b["topic"] = names[b["topic_id"]]
                if got.get((n, "block", i)):
                    b["summary"] = got[(n, "block", i)]
            intro = got.get((n, "intro", 0), "").strip()
            if not intro:
                sp = f" with {len(res['speakers'])} speakers ({', '.join(res['speakers'][:4])})" if res["speakers"] else ""
                intro = f"This document is a transcript of about {res['words']:,} words{sp}, covering {len(res['topics'])} main topics."
            res["overview"] = overview_md(intro, res["topics"])
            res["name"] = n
            self.results[n] = res

    def _save(self) -> None:
        d = OUT / time.strftime("%Y%m%d-%H%M%S")
        d.mkdir(parents=True, exist_ok=True)
        self.out_dir = str(d)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for n, r in self.results.items():
                md = to_markdown(r)
                (d / f"{safe(n)}.md").write_text(md, encoding="utf-8")
                z.writestr(f"{safe(n)}.md", md)
        self.zip = buf.getvalue()
        meta = dict(embedder=self.embedder.name, llm=self.llm.name if self.llm else "none (extractive)", seconds=round(self.elapsed, 1))
        (d / "results.json").write_text(json.dumps({"meta": meta, "results": self.results}), encoding="utf-8")

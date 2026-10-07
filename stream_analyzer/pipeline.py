"""Analyse ONE document: segments -> topics -> timeline (no AI model needed here)."""
from __future__ import annotations

from typing import List

import numpy as np
from sklearn.preprocessing import normalize

from .segmentation import cluster, find_segments, windows
from .text import SAFETY_RE, class_keywords, compress, dialogue, range_label, speaker_tokens, top_sents

def analyze_one(turns: List[dict], E: np.ndarray, cfg: dict):
    wc = np.array([len(t["text"].split()) for t in turns])
    F, B, C = windows(E, wc, cfg["window"])
    segs = find_segments(F, B, cfg["min_turns"], cfg["max_turns"], cfg["sens"])
    S = normalize(np.array([C[e] - C[s] for s, e in segs]))
    sw = np.array([wc[s:e].sum() for s, e in segs], float)
    lab = cluster(S, cfg["max_topics"], sw)
    stop = speaker_tokens(turns)
    texts = [" ".join(t["text"] for t in turns[s:e]) for s, e in segs]
    ids = sorted(set(lab.tolist()))
    grouped = {i: " ".join(x for x, lb in zip(texts, lab) if lb == i) for i in ids}
    tkw = dict(zip(ids, class_keywords([grouped[i] for i in ids], stop, 4)))
    skw = class_keywords(texts, stop, 4)
    segments = []
    for i, (s, e) in enumerate(segs):
        dlg = dialogue(turns[s:e])
        hits = sorted({m.lower() for m in SAFETY_RE.findall(dlg)})
        a, b = range_label(turns, s, e)
        segments.append(dict(segment=i + 1, start=a, end=b, turn_start=s + 1, turn_end=e, words=int(sw[i]),
                             topic_id=int(lab[i]), keywords=skw[i], summary=" ".join(top_sents(texts[i], 2)) or texts[i][:300],
                             safety="PASS" if not hits else "FLAGGED: " + ", ".join(hits), text=dlg))
    total = max(1.0, sw.sum())
    topics = []
    for tid in ids:
        mine = [x for x in segments if x["topic_id"] == tid]
        kw = tkw[tid]
        topics.append(dict(topic_id=tid, title=", ".join(kw[:3]).title() or "General discussion", description="", keywords=kw,
                           share_pct=round(100 * sum(x["words"] for x in mine) / total, 1), segments=len(mine),
                           when="; ".join(f"{x['start']} → {x['end']}" for x in mine)))
    timeline, block_text = [], []
    for x, t in zip(segments, texts):
        if timeline and timeline[-1]["topic_id"] == x["topic_id"]:
            b = timeline[-1]
            b["end"], b["words"] = x["end"], b["words"] + x["words"]
            b["summary"] += " " + x["summary"]
            b["keywords"] = list(dict.fromkeys(b["keywords"] + x["keywords"]))[:6]
            block_text[-1] += " " + t
        else:
            timeline.append({k: x[k] for k in ("start", "end", "topic_id", "keywords", "summary", "words")})
            block_text.append(t)
    for b in timeline:
        b["summary"] = " ".join(top_sents(b["summary"], 3)) or b["summary"]
    res = dict(n_turns=len(turns), words=int(wc.sum()), speakers=sorted({t["speaker"] for t in turns if t["speaker"]}),
               has_timestamps=any(t["ts"] for t in turns), overview="", topics=topics, timeline=timeline, segments=segments)
    excerpts = {"topics": {i: compress(grouped[i], 2400) for i in ids}, "blocks": [compress(t, 1500) for t in block_text]}
    return res, excerpts

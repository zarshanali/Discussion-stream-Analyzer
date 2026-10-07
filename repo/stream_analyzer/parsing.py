"""Raw transcript text -> list of turns {ts, speaker, text}."""
from __future__ import annotations

import re
from collections import Counter
from typing import List

TS = r"\d{1,2}:\d{2}(?::\d{2})?"
RX_TS = re.compile(rf"^[\[(]?({TS})(?:\.\d+)?(?:\s*(?:-->|-|–)\s*{TS}(?:\.\d+)?)?[\])]?\s*[-–]?\s*")
RX_SPK = re.compile(r"^(\w[\w.'-]*(?: \w[\w.'-]*){0,2})\s*:\s+(.*)$")
RX_SPK_TS = re.compile(rf"^(\w[\w.'-]*(?: \w[\w.'-]*){{0,2}})\s*[\[(]({TS})[\])]\s*:?\s*(.*)$")


def decode_bytes(b: bytes) -> str:
    for enc in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            return b.decode(enc)
        except Exception:
            continue
    return b.decode("utf-8", errors="ignore")


def split_long(turns: List[dict], cap: int = 70) -> List[dict]:
    """Long paragraphs become ~cap-word chunks, so a transcript with few lines still has topic boundaries to find."""
    out = []
    for t in turns:
        if len(t["text"].split()) <= cap:
            out.append(t)
            continue
        buf, n = [], 0
        for s in re.split(r"(?<=[.!?])\s+", t["text"]):
            w = len(s.split())
            if buf and n + w > cap:
                out.append({**t, "text": " ".join(buf)})
                buf, n = [], 0
            buf.append(s)
            n += w
        if buf:
            out.append({**t, "text": " ".join(buf)})
    return out


def parse(raw: str) -> List[dict]:
    """[00:01:23] Alice: text | Alice: text | 00:01:23 text | plain lines  ->  [{ts, speaker, text}]."""
    rows = []
    for line in raw.replace("\r", "").split("\n"):
        line = line.strip()
        if not line:
            continue
        ts, spk, text = None, None, line
        m = RX_TS.match(line)
        if m:
            ts, text = m.group(1), line[m.end():]
        m = RX_SPK.match(text)
        if m:
            spk, text = m.group(1), m.group(2)
        elif ts is None and (m := RX_SPK_TS.match(text)):
            spk, ts, text = m.group(1), m.group(2), m.group(3)
        rows.append([ts, spk, text.strip()])
    # a label is a speaker if it repeats, looks like a name, or is speaker_0 / Host / Guest ... ("The point is: ..." is not)
    known = re.compile(r"(speaker|spk|host|guest|interviewer|interviewee|moderator|narrator|unknown)[\w ]*$", re.I)
    name_like = lambda k: len(k.split()) <= 2 and all(w[0].isupper() for w in k.split())          # "Bob", "Dr. Smith"  # noqa: E731
    good = {k for k, v in Counter(r[1] for r in rows if r[1]).items() if v >= 2 or known.match(k) or name_like(k)}
    if sum(r[1] in good for r in rows) < 0.3 * max(1, len(rows)):
        good = set()
    turns: List[dict] = []
    for ts, spk, text in rows:
        if spk not in good:
            text, spk = (text if spk is None else f"{spk}: {text}"), None
        if spk is None and ts is None and turns and good:
            turns[-1]["text"] += " " + text                      # continuation line
        elif text:
            turns.append({"ts": ts, "speaker": spk, "text": text})
    return split_long(turns)

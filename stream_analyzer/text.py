"""Stop words, cleaning, key sentences and c-TF-IDF keywords."""
from __future__ import annotations

import re
from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer

FILLERS = set("""yeah yes yep nope okay ok um uh hmm mhm oh ah like just know think going gonna wanna right really actually
well sure thing things got get gets kind sort mean guess maybe lot little bit let lets say said says one two also want need
make makes made good great nice cool fine alright thanks thank hi hello hey bye probably definitely basically stuff something
anything everything everyone someone today time way back come came go went see look dont doesnt didnt thats its ill ive youre
were cant wont isnt wasnt theres youll weve theyre im hes shes literally pretty quite still even much many can could would
should will""".split())
STOP = set(ENGLISH_STOP_WORDS) | FILLERS
SAFETY_RE = re.compile(r"\b(confidential|secret key|password|api_key|api key|leak|private key)\b", re.IGNORECASE)


def clean(text: str) -> str:
    return re.sub(r"[’']", "", text)


def speaker_tokens(turns: List[dict]) -> set:
    return {w.lower() for t in turns if t["speaker"] for w in re.findall(r"[A-Za-z]{2,}", t["speaker"])}


def dialogue(turns: List[dict]) -> str:
    return "\n".join(f"{t['speaker']}: {t['text']}" if t["speaker"] else t["text"] for t in turns)


def range_label(turns, s, e) -> Tuple[str, str]:
    first = next((t["ts"] for t in turns[s:e] if t["ts"]), None)
    last = next((t["ts"] for t in reversed(turns[s:e]) if t["ts"]), None)
    return first or f"turn {s + 1}", last or f"turn {e}"


def rank_sents(text: str):
    sents = [s.strip() for line in text.split("\n") for s in re.split(r"(?<=[.!?])\s+", line) if len(s.split()) >= 5]
    sents = list(dict.fromkeys(sents))
    if len(sents) <= 2:
        return sents, list(range(len(sents)))
    try:
        X = TfidfVectorizer(stop_words=list(STOP)).fit_transform(sents)
        scores = np.asarray(X @ np.asarray(X.mean(axis=0)).T).ravel()          # closeness to the centroid
        return sents, list(np.argsort(-scores))
    except ValueError:
        return sents, list(range(len(sents)))


def top_sents(text: str, n: int) -> List[str]:
    sents, order = rank_sents(text)
    return [sents[i] for i in sorted(order[:n])]


def compress(text: str, max_chars: int) -> str:
    """Keep the most central sentences (in original order) until max_chars - the LLM sees a short, covering excerpt."""
    if len(text) <= max_chars:
        return text
    sents, order = rank_sents(text)
    keep, used = [], 0
    for i in order:
        if used + len(sents[i]) > max_chars:
            continue
        keep.append(i)
        used += len(sents[i]) + 1
    return " ".join(sents[i] for i in sorted(keep)) or text[:max_chars]


def _analyzer(stop: set):
    def analyze(doc: str) -> List[str]:
        toks = re.findall(r"[a-z]+", doc.lower())
        ok = [len(t) >= 3 and t not in stop for t in toks]
        out = []
        for i, t in enumerate(toks):
            if ok[i]:
                out.append(t)
                if i + 1 < len(toks) and ok[i + 1]:
                    out.append(t + " " + toks[i + 1])
        return out
    return analyze


def class_keywords(texts: List[str], extra_stop: set, top_n: int = 4) -> List[List[str]]:
    """c-TF-IDF (the idea behind BERTopic): terms frequent in one class but rare in the others."""
    try:
        cv = CountVectorizer(analyzer=_analyzer(STOP | extra_stop))
        X = cv.fit_transform([clean(t) for t in texts]).toarray().astype(float)
    except ValueError:
        return [[] for _ in texts]
    vocab, total = np.array(cv.get_feature_names_out()), X.sum(0)
    if X.sum() > 300:
        X = X * (total >= 2)
    score = X / np.maximum(X.sum(1, keepdims=True), 1) * np.log(1 + X.sum() / len(texts) / np.maximum(total, 1))
    out = []
    for row in score:
        chosen, used = [], set()
        for i in np.argsort(-row):
            if row[i] <= 0 or len(chosen) >= top_n:
                break
            words = set(vocab[i].split())
            if not words & used:
                chosen.append(vocab[i])
                used |= words
        out.append(chosen)
    return out

"""Where does the subject change (segmentation) and which segments belong together (clustering)."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

def windows(E: np.ndarray, wc: np.ndarray, target: int = 60, cap: int = 10):
    """Forward / backward window vectors for EVERY gap from the turn embeddings, using cumulative sums (no extra encoding)."""
    n = len(E)
    C = np.vstack([np.zeros((1, E.shape[1])), np.cumsum(E * np.sqrt(np.maximum(wc, 1))[:, None], 0)])
    cw = np.concatenate([[0], np.cumsum(wc)])
    i = np.arange(n)
    end = np.clip(np.searchsorted(cw, cw[i] + target), i + 1, np.minimum(i + cap, n))
    beg = np.clip(np.searchsorted(cw, cw[i] - target, side="right") - 1, np.maximum(i - cap, 0), np.maximum(i - 1, 0))
    F, B = normalize(C[end] - C[i]), normalize(C[i] - C[beg])
    B[0] = F[0]
    return F, B, C


def find_segments(F, B, min_turns: int, max_turns: int, sens: int) -> List[Tuple[int, int]]:
    """TextTiling-style: compare the text before and after every gap; deep similarity valleys become boundaries."""
    n = len(F)
    sims: Dict[int, float] = {}
    bounds: List[int] = []
    if n >= 2 * min_turns:
        bs = list(range(1, n))
        raw = (B[1:] * F[1:]).sum(1)
        sm = np.convolve(np.pad(raw, (1, 1), mode="edge"), np.ones(3) / 3, mode="valid")
        sims = dict(zip(bs, sm))
        depth = np.zeros(len(sm))
        for j in range(len(sm)):
            lo = j
            while lo > 0 and sm[lo - 1] >= sm[lo]:
                lo -= 1
            hi = j
            while hi < len(sm) - 1 and sm[hi + 1] >= sm[hi]:
                hi += 1
            depth[j] = (sm[lo] - sm[j]) + (sm[hi] - sm[j])
        cand = [j for j in range(len(sm)) if (j == 0 or sm[j] <= sm[j - 1]) and (j == len(sm) - 1 or sm[j] <= sm[j + 1])
                and min_turns <= bs[j] <= n - min_turns]
        if cand:
            d = depth[cand]
            cutoff = max(0.02, d.mean() + (5 - sens) * 0.3 * d.std())
            for j in sorted(cand, key=lambda j: -depth[j]):
                if depth[j] >= cutoff - 1e-9 and all(abs(bs[j] - x) >= min_turns for x in bounds):
                    bounds.append(bs[j])
    edges = [0] + sorted(bounds) + [n]

    def cut(s: int, e: int) -> List[Tuple[int, int]]:
        if e - s <= max_turns:
            return [(s, e)]
        options = list(range(s + min_turns, e - min_turns + 1)) or [(s + e) // 2]
        b = min(options, key=lambda b: (sims.get(b, 1.0), abs(b - (s + e) / 2)))
        return cut(s, b) + cut(b, e)

    return [x for s, e in zip(edges[:-1], edges[1:]) for x in cut(s, e)]


def relabel(lab: np.ndarray) -> np.ndarray:
    order = {old: new for new, old in enumerate(dict.fromkeys(lab.tolist()))}
    return np.array([order[x] for x in lab])


def cluster(E: np.ndarray, max_k: int, w: np.ndarray) -> np.ndarray:
    """Average-linkage clustering on cosine distance, k chosen by silhouette; tiny topics are merged into their neighbour."""
    n = len(E)
    if n <= 2:
        return np.zeros(n, dtype=int) if n < 2 or float(E[0] @ E[1]) > 0.7 else np.array([0, 1])
    D = np.clip(1 - E @ E.T, 0, 2)
    np.fill_diagonal(D, 0)
    best, best_score = np.zeros(n, dtype=int), -9.0
    for k in range(2, min(max_k, n - 1) + 1):
        lab = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average").fit_predict(D)
        if len(set(lab)) < 2:
            continue
        score = silhouette_score(D, lab, metric="precomputed") - 0.01 * k
        if score > best_score:
            best, best_score = lab, score
    if best_score < 0 and n <= 6:
        best = np.arange(n)
    lab = best.copy()
    while len(set(lab)) > 2:                                   # absorb topics below 6% of the words
        share = {i: w[lab == i].sum() / w.sum() for i in set(lab)}
        small = min(share, key=share.get)
        if share[small] >= 0.06:
            break
        v = normalize(E[lab == small].sum(0, keepdims=True))[0]
        cents = {i: normalize(E[lab == i].sum(0, keepdims=True))[0] for i in share if i != small}
        lab[lab == small] = max(cents, key=lambda i: cents[i] @ v)
    return relabel(lab)

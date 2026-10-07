import time

import numpy as np

import stream_analyzer.job as job_mod
from stream_analyzer.embeddings import Embedder
from stream_analyzer.job import Job
from stream_analyzer.segmentation import cluster, find_segments, windows


def run_job(docs, cfg):
    j = Job(docs, cfg, Embedder("x", use_transformer=False), None)
    for _ in range(300):
        if j.done:
            break
        time.sleep(0.1)
    assert j.done
    return j


def test_end_to_end(tmp_path, monkeypatch, docs, cfg):
    monkeypatch.setattr(job_mod, "OUT", tmp_path)
    j = run_job(docs, cfg)
    assert j.error == ""
    status = {f["name"]: f["status"] for f in j.snapshot()}
    assert status == {"ep1": "done", "ep2": "done", "plain": "done", "empty": "failed"}     # one bad file never stops the others
    r = j.results["ep1"]
    assert len(r["topics"]) >= 2 and r["overview"].startswith("This document is a transcript of")
    assert [b["start"] for b in r["timeline"]] == sorted(b["start"] for b in r["timeline"])
    assert (tmp_path / next(p.name for p in tmp_path.iterdir()) / "results.json").exists() and j.zip


def test_topic_boundaries_found(docs, cfg):
    from stream_analyzer.parsing import parse
    turns = parse(docs["ep1"])
    E = Embedder("x", use_transformer=False).encode([t["text"] for t in turns])
    wc = np.array([len(t["text"].split()) for t in turns])
    F, B, _ = windows(E, wc, 60)
    segs = find_segments(F, B, 5, 40, 6)
    assert segs[0][0] == 0 and segs[-1][1] == len(turns)
    assert all(a[1] == b[0] for a, b in zip(segs, segs[1:]))                                  # no gaps, no overlaps


def test_cluster_edge_cases():
    w = np.ones(3)
    assert list(cluster(np.ones((1, 4)), 6, w[:1])) == [0]
    assert len(cluster(np.eye(4)[:2], 6, w[:2])) == 2
    assert len(set(cluster(np.vstack([np.eye(3)] * 2), 6, np.ones(6)))) >= 1

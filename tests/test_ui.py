import time
from pathlib import Path

from streamlit.testing.v1 import AppTest

import stream_analyzer.job as job_mod
from stream_analyzer.embeddings import Embedder
from stream_analyzer.job import Job
from stream_analyzer.ui import registry

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_page_renders_idle_and_with_results(tmp_path, monkeypatch, docs, cfg):
    monkeypatch.setattr(job_mod, "OUT", tmp_path)
    registry().update(job=None, loaded=None)
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception

    j = Job(docs, cfg, Embedder("x", use_transformer=False), None)
    while not j.done:
        time.sleep(0.1)
    registry().update(job=j, loaded=None)
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    picker = next(s for s in at.selectbox if s.label == "Document")
    for name in ["All documents", "ep1", "ep2", "plain"]:
        picker.select(name)
        at.run()
        assert not at.exception, name
    registry().update(job=None, loaded=None)

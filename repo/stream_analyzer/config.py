"""All settings in one place."""
from pathlib import Path

OUT = Path("outputs")            # every run is saved in outputs/<timestamp>/
MAX_DOCS = 30                    # most transcripts per run

EMBEDDERS = {"Fast · MiniLM-L6": "sentence-transformers/all-MiniLM-L6-v2", "Better · BGE-small": "BAAI/bge-small-en-v1.5"}
LLMS = {"Fast · Qwen2.5-1.5B": "Qwen/Qwen2.5-1.5B-Instruct", "Better · Qwen2.5-3B": "Qwen/Qwen2.5-3B-Instruct",
        "No AI model · instant": None}
PALETTE = ["#4f46e5", "#e11d48", "#059669", "#d97706", "#7c3aed", "#0891b2", "#db2777", "#65a30d", "#0ea5e9", "#a16207"]

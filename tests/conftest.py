import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TOPICS = {
    "shoes": "The shoe repairman explained how resoling a leather boot works, the glue, the stitching, the cobbler workshop and new heels.",
    "money": "They discussed the economics of repair versus replacement, the cost of cheap sneakers, wages, profit margins and spending.",
    "family": "He told a story about his father, the family business, growing up in the neighborhood and training his apprentice daughter.",
}


def make_transcript(order, speakers=True, timestamps=True, seed=1):
    rnd, t, lines = random.Random(seed), 0, []
    for k in order:
        for i in range(14):
            t += rnd.randint(5, 40)
            who = "Host" if i % 2 == 0 else "Guest"
            body = f"{TOPICS[k]} Point {i} about {k} and more details on {k} matters."
            stamp = f"[{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}] " if timestamps else ""
            lines.append(f"{stamp}{who}: {body}" if speakers else body)
    return "\n".join(lines)


@pytest.fixture
def cfg():
    return dict(embedder="x", llm=None, ai_blocks=False, max_topics=6, min_turns=5, max_turns=40, sens=6, window=60)


@pytest.fixture
def docs():
    return {
        "ep1": make_transcript(["shoes", "money", "family", "shoes"]),
        "ep2": make_transcript(["money", "family"], timestamps=False),
        "plain": " ".join(TOPICS["shoes"] + f" detail {i}." for i in range(40)),
        "empty": "   \n  ",
    }

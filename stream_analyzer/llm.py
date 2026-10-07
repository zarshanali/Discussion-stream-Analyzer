"""The instruction-tuned LLM (Qwen) that writes topic titles, descriptions and the overview paragraph."""
from __future__ import annotations

import re
from typing import List, Tuple

SYSTEM = "You are a precise assistant that summarises transcripts. Use only facts stated in the text. Never invent names, numbers or places."
TOPIC_P = ("Below are excerpts of a transcript that cover the same theme. Write a short title (3-7 words, Title Case) and a "
           "description of 1-3 sentences with the key facts, keeping names, places and numbers. Third person, no bullet "
           "points. Answer in exactly this format:\nTITLE: ...\nDESCRIPTION: ...\n\nEXCERPTS:\n{}")
INTRO_P = ("Below is the beginning of a transcript and its main keywords. Write ONE paragraph (2-3 sentences) that starts with "
           "'This document is a transcript of' and says what kind of recording it is (podcast episode, meeting, interview, "
           "lecture...), names the show, the host and any guests if they are mentioned, and states the main subject. Use only "
           "facts present in the text.\n\nBEGINNING OF TRANSCRIPT:\n{}\n\nKEYWORDS: {}")
BLOCK_P = "Summarise this part of a transcript in 1-2 sentences. Keep names, places and numbers. Third person, plain prose.\n\n{}"


class LLM:
    def __init__(self, name: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if not torch.cuda.is_available():
            raise RuntimeError("No GPU found (Colab: Runtime > Change runtime type > T4 GPU).")
        self.name, self.torch = name, torch
        self.tok = AutoTokenizer.from_pretrained(name, padding_side="left")
        try:
            m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16)
        except TypeError:                                    # older transformers
            m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16)
        self.model = m.to("cuda").eval()

    def _run(self, prompts: List[str], max_new: int) -> List[str]:
        texts = [self.tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}],
                                              tokenize=False, add_generation_prompt=True) for p in prompts]
        enc = self.tok(texts, return_tensors="pt", padding=True).to(self.model.device)
        try:
            with self.torch.inference_mode():
                g = self.model.generate(**enc, max_new_tokens=max_new, do_sample=False, repetition_penalty=1.05,
                                        pad_token_id=self.tok.pad_token_id)
        except self.torch.cuda.OutOfMemoryError:             # split the batch and try again
            self.torch.cuda.empty_cache()
            if len(prompts) == 1:
                return [""]
            h = len(prompts) // 2
            return self._run(prompts[:h], max_new) + self._run(prompts[h:], max_new)
        return [self.tok.decode(s[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip() for s in g]

    def generate(self, prompts: List[str], max_new: int = 170, token_budget: int = 12000, progress=None, stop=None) -> List[str]:
        """Longest prompts first, grouped into batches of similar length (same idea as the speech-to-text app)."""
        order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
        out, a = [""] * len(prompts), 0
        while a < len(order):
            if stop and stop():
                break
            longest = len(prompts[order[a]]) / 3.5 + 150
            b = a + max(1, min(16, int(token_budget // longest)))
            idx = order[a:b]
            for i, text in zip(idx, self._run([prompts[i] for i in idx], max_new)):
                out[i] = text
            a = b
            if progress:
                progress(a / len(order))
        return out


def title_desc(raw: str) -> Tuple[str, str]:
    t, d = re.search(r"TITLE:\s*(.+)", raw), re.search(r"DESCRIPTION:\s*(.+)", raw, re.S)
    title = t.group(1).strip().strip("*# ") if t else ""
    desc = d.group(1).strip() if d else re.sub(r"TITLE:.*\n?", "", raw).strip()
    return title, " ".join(desc.split())


def overview_md(intro: str, topics: List[dict]) -> str:
    return intro.strip() + "\n\n**Key Topics Discussed:**\n\n" + "\n\n".join(f"* **{t['title']}:** {t['description']}" for t in topics)

from stream_analyzer.parsing import decode_bytes, parse


def test_speakers_and_timestamps():
    turns = parse("[00:01:23] Alice: hello there\n[00:01:30] Bob: hi\n[00:01:40] Alice: again")
    assert [t["speaker"] for t in turns] == ["Alice", "Bob", "Alice"]
    assert turns[0]["ts"] == "00:01:23"


def test_sentence_with_colon_is_not_a_speaker():
    turns = parse("So here is the thing: we left.\nAlice: hello there\nBob: hi\nAlice: again fine\n")
    assert turns[0]["speaker"] is None and turns[0]["text"].startswith("So here is the thing")


def test_speech_to_text_app_format():
    turns = parse("[00:00:01.00 - 00:00:03.00] speaker_0: hello world friends\n[00:00:04.00 - 00:00:06.00] speaker_1: hi\n")
    assert [t["speaker"] for t in turns] == ["speaker_0", "speaker_1"] and turns[0]["ts"] == "00:00:01"


def test_long_paragraph_is_split():
    assert len(parse(" ".join(f"This is sentence number {i} of a long paragraph." for i in range(60)))) > 3


def test_decode_bytes():
    assert decode_bytes("héllo".encode("utf-16")) == "héllo"
    assert decode_bytes("héllo".encode("latin-1")) == "héllo"

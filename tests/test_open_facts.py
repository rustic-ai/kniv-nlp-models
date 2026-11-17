"""Schema-free facts decoded from a hand-built linguistic layer."""
from kniv_v7.open_facts import decode


def _w(pos, lemma, head, rel):
    return [{"pos": p, "lemma": l, "head": h, "deprel": r} for p, l, h, r in zip(pos, lemma, head, rel)]


def test_frame_with_preposition_time_and_negation():
    # Microsoft did not acquire GitHub in Texas in 2018 .
    toks = ["Microsoft", "did", "not", "acquire", "GitHub", "in", "Texas", "in", "2018", "."]
    a = {"tokens": toks,
         "words": _w(["PROPN", "AUX", "PART", "VERB", "PROPN", "ADP", "PROPN", "ADP", "NUM", "PUNCT"],
                     ["Microsoft", "do", "not", "acquire", "GitHub", "in", "Texas", "in", "2018", "."],
                     [4, 4, 4, 0, 4, 7, 4, 9, 4, 4],
                     ["nsubj", "aux", "advmod", "root", "obj", "case", "obl", "case", "obl", "punct"]),
         "srl": [{"predicate": 3, "args": [["ARG0", 0, 1], ["ARGM-NEG", 2, 3], ["ARG1", 4, 5],
                                           ["ARGM-LOC", 5, 7], ["ARGM-TMP", 7, 9]]}],
         "coref": []}
    got = {(f["subject"]["text"], f["predicate"]["text"], f["object"]["text"]): f for f in decode(a)}
    assert set(got) == {("Microsoft", "acquire", "GitHub"), ("Microsoft", "acquire in", "Texas")}
    f = got[("Microsoft", "acquire", "GitHub")]
    assert f["polarity"] == "negative" and f["time"] == "in 2018" and f["source"] == "srl:ARG1"


def test_conjunct_cut_pronoun_dropped_and_copula():
    # We hired Ann and Bob . Ann is the chief executive of Acme .
    toks = ["We", "hired", "Ann", "and", "Bob", ".", "Ann", "is", "the", "chief", "executive", "of", "Acme", "."]
    a = {"tokens": toks,
         "words": _w(["PRON", "VERB", "PROPN", "CCONJ", "PROPN", "PUNCT",
                      "PROPN", "AUX", "DET", "ADJ", "NOUN", "ADP", "PROPN", "PUNCT"],
                     ["we", "hire", "Ann", "and", "Bob", ".", "Ann", "be", "the", "chief", "executive", "of", "Acme", "."],
                     [2, 0, 2, 5, 3, 2, 11, 11, 11, 11, 0, 13, 11, 11],
                     ["nsubj", "root", "obj", "cc", "conj", "punct",
                      "nsubj", "cop", "det", "amod", "root", "case", "nmod", "punct"]),
         "srl": [{"predicate": 1, "args": [["ARG0", 0, 1], ["ARG1", 2, 5]]}],
         "coref": []}
    got = {(f["subject"]["text"], f["predicate"]["text"], f["object"]["text"]): f for f in decode(a)}
    # "We" has no cluster: the hiring fact is dropped
    assert set(got) == {("Ann", "is", "the chief executive of Acme")}
    assert got[("Ann", "is", "the chief executive of Acme")]["predicate"]["phrase"] == "is the chief executive of"


def test_conversation_speaker_and_holder():
    # I live in Berlin . (speaker: user)
    toks = ["I", "live", "in", "Berlin", "."]
    a = {"tokens": toks,
         "words": _w(["PRON", "VERB", "ADP", "PROPN", "PUNCT"], ["I", "live", "in", "Berlin", "."],
                     [2, 0, 4, 2, 2], ["nsubj", "root", "case", "obl", "punct"]),
         "srl": [{"predicate": 1, "args": [["ARG0", 0, 1], ["ARGM-LOC", 2, 4]]}],
         "coref": [], "sentences": [{"start": 0, "end": 5, "speaker": "user"}]}
    (f,) = decode(a)
    assert (f["subject"]["text"], f["predicate"]["text"], f["object"]["text"]) == ("user", "live in", "Berlin")
    assert f["holder"] == "user"

"""naive tier: WordNet same-POS synonym substitution on content words.

~15% substitution rate by default. No word-sense disambiguation (intentional
— keeps the tier "naive" and is a deliberate, documented limitation: it can
produce context-inappropriate synonyms, e.g. "artificial" -> "hokey".
"""
import random
import re

import nltk
from nltk.corpus import stopwords, wordnet

_NLTK_RESOURCES = [
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/stopwords", "stopwords"),
]

_initialized = False


def _ensure_resources():
    global _initialized
    if _initialized:
        return
    for find_path, download_name in _NLTK_RESOURCES:
        try:
            nltk.data.find(find_path)
        except LookupError:
            nltk.download(download_name, quiet=True)
    _initialized = True


_CONTENT_POS_PREFIXES = ("NN", "VB", "JJ", "RB")  # noun, verb, adj, adverb


def _wn_pos(tag: str):
    if tag.startswith("NN"):
        return wordnet.NOUN
    if tag.startswith("VB"):
        return wordnet.VERB
    if tag.startswith("JJ"):
        return wordnet.ADJ
    if tag.startswith("RB"):
        return wordnet.ADV
    return None


def _synonyms_for(word: str, pos: str) -> list[str]:
    """Return candidate synonyms for `word` with WordNet POS `pos`.

    Factored out so tests can monkeypatch/inject a deterministic map without
    requiring the WordNet corpus to be downloaded.
    """
    wn_pos = _wn_pos(pos)
    if wn_pos is None:
        return []
    candidates = set()
    for syn in wordnet.synsets(word, pos=wn_pos):
        for lemma in syn.lemmas():
            name = lemma.name().replace("_", " ")
            if name.lower() != word.lower():
                candidates.add(name)
    return list(candidates)


def _match_case(source: str, target: str) -> str:
    if source.isupper():
        return target.upper()
    if source[0].isupper():
        return target[0].upper() + target[1:]
    return target


def humanize(text: str, substitution_rate: float = 0.15, seed: int | None = None,
             dry_run: bool = False) -> str:
    if dry_run:
        return text
    _ensure_resources()
    rng = random.Random(seed)  # seed=None -> non-deterministic, matches config.yaml default

    stop_words = set(stopwords.words("english"))
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)

    # Candidate indices: content words, not stopwords, alphabetic.
    candidate_idx = [
        i for i, (word, tag) in enumerate(tagged)
        if tag.startswith(_CONTENT_POS_PREFIXES)
        and word.lower() not in stop_words
        and word.isalpha()
    ]
    n_to_replace = max(0, round(len(candidate_idx) * substitution_rate))
    chosen = set(rng.sample(candidate_idx, min(n_to_replace, len(candidate_idx))))

    out_tokens = []
    for i, (word, tag) in enumerate(tagged):
        if i in chosen:
            syns = _synonyms_for(word, tag)
            if syns:
                replacement = rng.choice(syns)
                word = _match_case(word, replacement)
        out_tokens.append(word)

    # Reassemble with reasonable spacing (no space before punctuation).
    out = ""
    for i, tok in enumerate(out_tokens):
        if i > 0 and not re.match(r"^[.,!?;:')\]]", tok):
            out += " "
        out += tok
    return out

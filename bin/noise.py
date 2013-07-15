from lexicon import word_hash
import sys

_indexed_weights = None

def indexed_weights():
    global _indexed_weights
    if _indexed_weights is not None:
        return _indexed_weights

    assert word_hash.len == VOCABULARY_SIZE
    _indexed_weights = [1 for id in range(word_hash.len)]
    return _indexed_weights

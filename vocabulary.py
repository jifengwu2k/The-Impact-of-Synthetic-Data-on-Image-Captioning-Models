from collections import Counter
from typing import Sequence

import nltk; nltk.download('punkt')


class Vocabulary:
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.idx2word: dict[int, str] = {
            0: '<<padding>>',
            1: '<<start>>',
            2: '<<end>>',
            3: '<<unknown>>'
        }

        self.word2idx: dict[str, int] = {
            '<<padding>>': 0,
            '<<start>>': 1,
            '<<end>>': 2,
            '<<unknown>>': 3
        }

        self.idx: int = 4

    def add_word(self, word: str) -> None:
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word: str) -> int:
        if not word in self.word2idx:
            return self.word2idx['<<unknown>>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def get_vocabulary_from_captions(captions: Sequence[str], word_frequency_count_threshold: int) -> Vocabulary:
    word_frequency_counter = Counter()

    for caption in captions:
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        word_frequency_counter.update(tokens)

    vocabulary = Vocabulary()

    for word, word_frequency_count in word_frequency_counter.items():
        if word_frequency_count >= word_frequency_count_threshold:
            vocabulary.add_word(word)

    return vocabulary

import typing

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


nltk.download('punkt')


def bleu_score(
        sampled_caption_tokens_sequence: typing.Sequence[str],
        ground_truth_captions_sequence: typing.Sequence[str],
):
    ground_truth_captions_tokens_list: list[list[str]] = [
        nltk.word_tokenize(ground_truth_caption.lower())
        for ground_truth_caption in ground_truth_captions_sequence
    ]

    generated_score = sentence_bleu(
        ground_truth_captions_tokens_list,
        sampled_caption_tokens_sequence,
        smoothing_function=SmoothingFunction().method4
    )

    theoretical_score = 0
    for i, ground_truth_caption_tokens in enumerate(ground_truth_captions_tokens_list):
        theoretical_score += sentence_bleu(
            ground_truth_captions_tokens_list[:i] + ground_truth_captions_tokens_list[i+1:],
            sampled_caption_tokens_sequence,
            smoothing_function=SmoothingFunction().method4
        )
    theoretical_score /= len(ground_truth_captions_tokens_list)

    return generated_score, theoretical_score

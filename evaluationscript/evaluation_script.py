"""
Evaluation script. Takes two files as input and compares the sentences using various metrics such as BLEU and METEOR.
"""
import os
import sys

import nltk
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score


def main():
    """Main function for the script.
    """
    # Download wordnet so that METEOR scorer works.
    nltk.download('wordnet')

    # Open truth.txt and answer.txt and ensure they have same number of lines.
    with open(sys.argv[1], 'r', encoding='utf8') as file_obj:
        true_sentences = file_obj.readlines()

    with open(sys.argv[2], 'r', encoding='utf8') as file_obj:
        pred_sentences = file_obj.readlines()

    if len(true_sentences) != len(pred_sentences):
        print(f'E: Number of sentences do not match. True: {len(true_sentences)} Pred: {len(pred_sentences)}')
        sys.exit()

    print(f'D: Number of sentences: {len(true_sentences)}')

    scores = {}

    # Macro-averaged BLEU-4 score.
    scores['bleu_4_macro'] = 0
    for ref, hyp in zip(true_sentences, pred_sentences):
        scores['bleu_4_macro'] += sentence_bleu(
            [ref.split()],
            hyp.split(),
            smoothing_function=SmoothingFunction().method2
        )
    scores['bleu_4_macro'] /= len(true_sentences)

    # BLEU-4 score.
    scores['bleu_4'] = corpus_bleu(
        [[ref.split()] for ref in true_sentences],
        [hyp.split() for hyp in pred_sentences],
        smoothing_function=SmoothingFunction().method2
    )

    # METEOR score.
    scores['meteor'] = 0
    for ref, hyp in zip(true_sentences, pred_sentences):
        scores['meteor'] += single_meteor_score(ref, hyp)
    scores['meteor'] /= len(true_sentences)

    print(f'D: Scores: {scores}')

    # Print out scores.
    for key in scores:
        print(f'{key}: {scores[key]}')


if __name__ == '__main__':
    main()

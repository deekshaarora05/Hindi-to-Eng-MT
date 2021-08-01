"""
Codalab evaluation script. Takes two files as input and compares the sentences using various metrics such as BLEU and
METEOR. Output is written to a file.
"""
import os
import re
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
    file1 = open("hypotheses.txt", 'r')
    pred_sentences = file1.readlines()
    file2 = open("references.txt", 'r')
    true_sentences = file2.readlines()

    true_s, pred_s = [], []
    for i in range(len(true_sentences)):
        true_s.append(true_sentences[i][0].lower())
        pred_s.append(pred_sentences[i][0].lower())

    true_sentences, pred_sentences = [], []
    true_sentences_joined, pred_sentences_joined = [], []

    for i in range(len(true_s)):
        # some punctuations from string.punctuation
        split_true = list(filter(None, re.split(r'[\s!"#$%&\()+,-./:;<=>?@\\^_`{|}~]+', true_s[i])))
        split_pred = list(filter(None, re.split(r'[\s!"#$%&\()+,-./:;<=>?@\\^_`{|}~]+', pred_s[i])))
        true_sentences.append(split_true)
        pred_sentences.append(split_pred)
        true_sentences_joined.append(' '.join(split_true))
        pred_sentences_joined.append(' '.join(split_pred))

    if len(true_sentences) != len(pred_sentences):
        print(f'E: Number of sentences do not match. True: {len(true_sentences)} Pred: {len(pred_sentences)}')
        sys.exit()

    print(f'D: Number of sentences: {len(true_sentences)}')

    scores = {}

    # Macro-averaged BLEU-4 score.
    scores['bleu_4_macro'] = 0
    for ref, hyp in zip(true_sentences, pred_sentences):
        scores['bleu_4_macro'] += sentence_bleu(
            [ref],
            hyp,
            smoothing_function=SmoothingFunction().method2
        )
    scores['bleu_4_macro'] /= len(true_sentences)

    # BLEU-4 score.
    scores['bleu_4'] = corpus_bleu(
        [[ref] for ref in true_sentences],
        [hyp for hyp in pred_sentences],
        smoothing_function=SmoothingFunction().method2
    )

    # METEOR score.
    scores['meteor'] = 0
    # changed
    for ref, hyp in zip(true_sentences_joined, pred_sentences_joined):
        scores['meteor'] += single_meteor_score(ref, hyp)
    scores['meteor'] /= len(true_s)

    print(f'D: Scores: {scores}')

    # Write scores to output file.
    # with open(os.path.join(sys.argv[2], 'scores.txt'), 'w', encoding='utf8') as file_obj:
    #     for key in scores:
    #         file_obj.write(f'{key}: {scores[key]}\n')
    #     file_obj.write('bleu_score: ' + str(scores['bleu_4']))


if _init_ == '_main_':
    main()
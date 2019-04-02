#!/usr/bin/python
# This scripts loads a pretrained model and a raw .txt files. It then performs sentence splitting and tokenization and passes
# the input sentences to the model for tagging. Prints the tokens and the tags in a CoNLL format to stdout

# It uses a particular tokenization scheme in line with the tokenization used to generate the training files (moses)

# Usage: python RunModel_modified.py modelPath inputPath
# For pretrained models see docs/Pretrained_Models.md
from __future__ import print_function
from neuralnets.BiLSTM import BiLSTM
from nltk.tokenize.regexp import RegexpTokenizer
from nltk import sent_tokenize, word_tokenize

from util.preprocessing import addCharInformation, createMatrices, addCasingInformation, addIsNameInformation, \
    load_names, FR_NAMES_PATH
import sys
import re

pattern = r"\@-\@|\w+['´`]|\w+|\S+"
regex_tokenizer = RegexpTokenizer(pattern, flags=re.UNICODE | re.IGNORECASE)


def pre_treat_text(raw_text):
    # TODO: We should really use the same tokenizer used to generate the train.iob file (moses)
    pre_treat_text = re.sub(r"(\w{2,})-(\w+)", r"\1@-@\2", raw_text, flags=re.IGNORECASE)  # Add @ to dashes
    pre_treat_text = re.sub(r"\n{2,}", r"\n", pre_treat_text)  # Replace two or more lines by a single line
    pre_treat_text = re.sub(r"\xa0", r" ", pre_treat_text)  # Replace this new line symbol by a space
    pre_treat_text = re.sub(r"_+", r"", pre_treat_text)  # Underscore kills Tagger training :/

    pre_treated_lines = pre_treat_text.split("\n")

    return pre_treated_lines, pre_treat_text


def tokenize_text(text_lines):
    sentences_tokens = []

    if not isinstance(text_lines, list):
        text_lines = [text_lines]

    for line in text_lines:
        sentences = sent_tokenize(line, language="french")
        for sentence in sentences:
            tokens = regex_tokenizer.tokenize(sentence)
            sentences_tokens.append(tokens)

    return sentences_tokens


def main():
    if len(sys.argv) < 3:
        print("Usage: python RunModel_modified.py modelPath inputPath")
        exit()

    modelPath = sys.argv[1]
    inputPath = sys.argv[2]

    # :: Read input ::
    with open(inputPath, 'r') as f:
        text = f.read()

    # :: Load vocabulary for is_name features ::
    from flashtext import KeywordProcessor
    keyword_processor = KeywordProcessor()
    keyword_processor.add_keywords_from_list(list(load_names(FR_NAMES_PATH).keys()))

    # :: Load the model ::
    lstmModel = BiLSTM.loadModel(modelPath)

    # :: Prepare the input ::
    pre_treated_lines, _ = pre_treat_text(text)
    tokenized_sentences = tokenize_text(pre_treated_lines)
    sentences = [{'tokens': sent} for sent in tokenized_sentences]
    addCharInformation(sentences)
    addCasingInformation(sentences)
    addIsNameInformation(sentences, keyword_processor=keyword_processor)
    dataMatrix = createMatrices(sentences, lstmModel.mappings, True)

    # :: Tag the input ::
    tags = lstmModel.tagSentences(dataMatrix)

    # :: Output to stdout ::
    for sentenceIdx in range(len(sentences)):
        tokens = sentences[sentenceIdx]['tokens']

        for tokenIdx in range(len(tokens)):
            tokenTags = []
            for modelName in sorted(tags.keys()):
                tokenTags.append(tags[modelName][sentenceIdx][tokenIdx])

            print("%s\t%s" % (tokens[tokenIdx], "\t".join(tokenTags)))
        print("")


if __name__ == '__main__':
    main()

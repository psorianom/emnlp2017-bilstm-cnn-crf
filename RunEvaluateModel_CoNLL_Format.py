#!/usr/bin/python
# This scripts loads a pretrained model and a input TEST file (with correct tags) in CoNLL format (each line a token, sentences separated by an empty line).
# The input sentences are passed to the model for tagging. Prints the tokens, the correct tags, and the predicted tags in a CoNLL format to stdout
# Usage: python RunModel_ConLL_Format.py modelPath inputPathToConllFile
# For pretrained models see docs/
from __future__ import print_function
from util.preprocessing import readCoNLL, createMatrices, addCharInformation, addCasingInformation
from neuralnets.BiLSTM import BiLSTM
import sys
import logging


if len(sys.argv) < 3:
    print("Usage: python RunModel_CoNLL_Format.py modelPath inputPathToConllFile")
    exit()

modelPath = sys.argv[1]
inputPath = sys.argv[2]
# inputColumns = {0: "tokens", 2: "NER_BIO"}
inputColumns = {0: "tokens", 1: "is_name", 2: "NER_BIO"}


# :: Prepare the input ::
sentences = readCoNLL(inputPath, inputColumns)
addCharInformation(sentences)
addCasingInformation(sentences)


# :: Load the model ::
lstmModel = BiLSTM.loadModel(modelPath)


dataMatrix = createMatrices(sentences, lstmModel.mappings, True)

# :: Tag the input ::
tags = lstmModel.tagSentences(dataMatrix)


# :: Output to stdout ::
all_sentences_preds = []
for sentenceIdx in range(len(sentences)):
    tokens = sentences[sentenceIdx]['tokens']
    correct_tag = sentences[sentenceIdx]['NER_BIO']
    for tokenIdx in range(len(tokens)):
        tokenTags = []
        for modelName in sorted(tags.keys()):
            tokenTags.append(correct_tag[tokenIdx])  # Predicted tag
            tokenTags.append(tags[modelName][sentenceIdx][tokenIdx])  # Correct tag
        string_temp = "%s %s" % (tokens[tokenIdx], " ".join(tokenTags))
        all_sentences_preds.append(string_temp)
        print(string_temp)
    all_sentences_preds.append(" ")
    print("")

# from util.conlleval import evaluate, report
# report(evaluate(all_sentences_preds))

#!/usr/bin/python
# This scripts loads a pretrained model and a input file in CoNLL format (each line a token, sentences separated by an empty line).
# The input sentences are passed to the model for tagging. Prints the tokens and the tags in a CoNLL format to stdout
# Usage: python RunModel_ConLL_Format.py modelPath inputPathToConllFile
# For pretrained models see docs/
from __future__ import print_function

import subprocess

from util.preprocessing import readCoNLL, createMatrices, addCharInformation, addCasingInformation
from neuralnets.BiLSTM import BiLSTM
import sys
import logging



if len(sys.argv) < 3:
    print("Usage: python RunModel_CoNLL_Format.py modelPath inputPath")
    exit()



modelPath = sys.argv[1]
inputPath = sys.argv[2]

# 1 create text file
subprocess.check_call(["python", "/home/pavel/code/conseil_detat/src/data/doc2txt.py", inputPath])
decision_txt_path = inputPath[:-3] + "txt"

# 2 file to conll file
subprocess.check_call(["python", "/home/pavel/code/conseil_detat/src/data/normal_doc2conll.py", decision_txt_path])


decision_conll_path = decision_txt_path[:-4] + "_CoNLL.txt"

#3 predict conll file



inputColumns = {0: "tokens"}


# :: Prepare the input ::
sentences = readCoNLL(decision_conll_path, inputColumns)
addCharInformation(sentences)
addCasingInformation(sentences)


# :: Load the model ::
lstmModel = BiLSTM.loadModel(modelPath)


dataMatrix = createMatrices(sentences, lstmModel.mappings, True)

# :: Tag the input ::
tags = lstmModel.tagSentences(dataMatrix)


# :: Output to stdout ::
list_token_tags = []
for sentenceIdx in range(len(sentences)):
    tokens = sentences[sentenceIdx]['tokens']
    sentence = []
    for tokenIdx in range(len(tokens)):

        tokenTags = []
        for modelName in sorted(tags.keys()):
            tokenTags.append(tags[modelName][sentenceIdx][tokenIdx])
        sentence.append((tokens[tokenIdx], tokenTags[0]))
        print("%s\t%s" % (tokens[tokenIdx], "\t".join(tokenTags)))
    print("")
    list_token_tags.append(sentence)

# print(list_token_tags)

anon_list_token_tags = []
for sent in list_token_tags:
    sentence = []
    for token, tag in sent:
        new_token = "..." if tag != "O" else token
        sentence.append((new_token, tag))
    anon_list_token_tags.append(sentence)
# print(anon_list_token_tags)

annon_file_path = decision_conll_path[:-10] + "_annon.txt"

with open(annon_file_path, "w") as filo:
    for sent in anon_list_token_tags:
        tokens = [t[0] for t in sent]
        filo.write(" ".join(tokens) + "\n")
        # print(tokens)
        # print()
        filo.write("\n")
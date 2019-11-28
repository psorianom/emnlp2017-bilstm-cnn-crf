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
inputColumns = {0: "tokens", 1: "NER_BIO"}
#inputColumns = {0: "tokens", 1: "is_name", 2: "NER_BIO"}


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

from util.conlleval import evaluate, report
report(evaluate(all_sentences_preds))


y_trues = [f.split()[1] for f in all_sentences_preds if f.lstrip()]
y_preds = [f.split()[2] for f in all_sentences_preds if f.lstrip()]
#
#y_trues = [f.split("-")[1] if f != "O" else f for f in y_trues]
#y_preds = [f.split("-")[1] if f != "O" else f for f in y_preds]
#
from sklearn.metrics import confusion_matrix
# #print(confusion_matrix(y_trues, y_preds, ["B-AUX", "I-AUX", "B-DATE", "I-DATE", "B-LOC", "I-LOC", "B-PER", "I-PER", "O"]))
print(confusion_matrix(y_trues, y_preds, ["B-PER_NOM", "I-PER_NOM", "B-PER_PRENOM", "I-PER_PRENOM", "B-LOC", "I-LOC", "O"]))
#
# df_cm = confusion_matrix(y_trues, y_preds, ["AUX", "DATE", "LOC", "PER", "O"])
df_cm = confusion_matrix(y_trues, y_preds, ["B-PER_NOM", "I-PER_NOM", "B-PER_PRENOM", "I-PER_PRENOM", "B-LOC", "I-LOC", "O"])
#
import pandas as pd
pd.set_option('display.max_columns', 500)
pd.set_option('expand_frame_repr', False)
# df_cm = pd.DataFrame(df_cm, index=["V_AUX", "V_DATE", "V_LOC", "V_PER", "V_O"], columns=["P_AUX", "P_DATE", "P_LOC", "P_PER", "P_O"])
df_cm = pd.DataFrame(df_cm, index=["V_B-PER_NOM", "V_I-PER_NOM", "V_B-PER_PRENOM", "V_I-PER_PRENOM", "V_B-LOC", "V_I-LOC", "V_O"],
                     columns=["P_B-PER_NOM", "P_I-PER_NOM", "P_B-PER_PRENOM", "P_I-PER_PRENOM", "P_B-LOC", "P_I-LOC", "P_O"])

print(df_cm)
# This script trains the BiLSTM-CNN-CRF architecture for Chunking in English using
# the CoNLL 2000 dataset (https://www.clips.uantwerpen.be/conll2000/chunking/).
# The code use the embeddings by Komninos et al. (https://www.cs.york.ac.uk/nlp/extvec/)
from __future__ import print_function
import os
import logging
import sys
from neuralnets.BiLSTM import BiLSTM
from util.preprocessing import perpareDataset, loadDatasetPickle

from keras import backend as K
K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=8, inter_op_parallelism_threads=8)))

# :: Change into the working dir of the script ::
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# :: Logging level ::
loggingLevel = logging.INFO
logger = logging.getLogger()
logger.setLevel(loggingLevel)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(loggingLevel)
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


######################################################
#
# Data preprocessing
#
######################################################
datasets = {
    'conseil_etat':                                   #Name of the dataset
         {'columns': {0:'tokens', 1:'NER_BIO'},
         # {'columns': {0:'tokens', 1:"is_name",  2:'NER_BIO'},   #CoNLL format for the input data. Column 0 contains tokens, column 1 contains POS and column 2 contains chunk information using BIO encoding
         'label': 'NER_BIO',                              #Which column we like to predict
         'evaluate': True,                                  #Should we evaluate on this task? Set true always for single task setups
         'commentSymbol': None,
         }
}

# :: Path on your computer to the word embeddings. Embeddings by Komninos et al. will be downloaded automatically ::
# embeddingsPath = '/home/pavel/data/embeddings/doctrine/embeddings.vec.tar.gz'
embeddingsPath = "embeddings.vec"

# :: Prepares the dataset to be used with the LSTM-network. Creates and stores cPickle files in the pkl/ folder ::
pickleFile = perpareDataset(embeddingsPath, datasets, useExistent=False)


######################################################
#
# The training of the network starts here
#
######################################################


#Load the embeddings and the dataset
embeddings, mappings, data = loadDatasetPickle(pickleFile)

# Some network hyperparameters
params = {'classifier': ['CRF'], 'LSTM-Size': [100, 100], 'dropout': (0.5, 0.5), 'charEmbeddings':'LSTM',
          'optimizer': 'adam', 'featureNames': ['tokens', 'casing']}


MODEL = BiLSTM(params)
MODEL.setMappings(mappings, embeddings)
MODEL.setDataset(datasets, data)
MODEL.storeResults('results/Conseil_NER.csv') #Path to store performance scores for dev / test
MODEL.modelSavePath = "models/[ModelName]_[DevScore]_[TestScore]_[Epoch].h5"
MODEL.fit(epochs=100)




#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 09:23:02 2017

@author: wevers
"""

import gensim, logging
import nltk
import os
import pandas as pd
import glob
import re
import argparse
import pprint
import gensim
from nltk import sent_tokenize
from string import punctuation

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

yearsInModel = 5
steYears = 1
modelFolder = 'tempModels'

y0 = 1970
yN = 1994
    

def getSentencesInRange(startY, endY):
    sentences = []
    for year in range(startY, endY):
        for file in glob.glob('data2/*):
            if year in file:
                append 

for year in range(y0, yN-yearsInModel+1, stepYears):
    startY = year
    endY = year + yearsInModel
    modelName = modelFolder + '/%d_%d.w2v'%(year,year+yearsInModel)
    print('Building Model: ', modelName)
    
    sentences = getSentencesInRange(startY, endY)

path = 'data2'
allFiles = glob.glob(path + "/*.tsv")




model = gensim.models.Word2Vec(sentences, alpha = 0.05, window=5, min_count=5, size = 140, workers=4)
model.init_sims(replace=True)
    #model.wv.save_word2vec_format("test", binary=True)
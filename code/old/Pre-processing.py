#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 14:09:38 2017

@author: melvinwevers
"""
import glob
import re
import pandas as pd
import nltk
import gensim, logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

tokenizer = nltk.data.load('tokenizers/punkt/dutch.pickle')

path = 'test'
path2 = 'test2'
allFiles = glob.glob(path + "/*.tsv")

def article_to_sentences(text):
    raw_sentences = tokenizer.tokenize(text.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences += cleaning(raw_sentence)
    return sentences

def cleaning(text):
    text = re.sub("[^a-zA-Z]"," ", text)
    words = text.lower().split()
    return(words)
    
def getSentencesForYear(year):
    for file_ in allFiles:
        filename = re.sub(path, '', file_)
        if filename.startswith(str(year)):
            df = pd.read_csv(file_, sep='\t', encoding='utf-8')
            df = df[df['ocr'].notnull()] #remove empty rows
            sentences = df['ocr'].apply(article_to_sentences)
            sentences = sentences.tolist()
            sentences = [item for sublist in sentences for item in sublist]
            #print(sentencesYear)
            #sentencesYear.to_csv(path2 + '/'+ 'xxsentences' + filename, index=False, sep='\t', encoding='utf-8')
    return sentences_

x_ = getSentencesForYear(1970)

def getSentencesInRange(startY, endY):
    return [s for year in range(startY, endY)
            for s in getSentencesForYear(year)]

#getSentencesInRange(1970,1971)

yearsInModel = 5    
stepYears = 1
modelFolder = 'tempModels'

y0 = 1975
yN = 1994

for year in range(y0, yN-yearsInModel+1, stepYears):
    startY = year
    endY = year + yearsInModel
    modelName = modelFolder + '/%d_%d.w2v'%(year,year+yearsInModel)
    print('Building Model: ', modelName)
    
    sentences_ = getSentencesInRange(startY, endY)
    
#    #model = gensim.models.Word2Vec(min_count=10, size = 200, iter=10, alpha=0.025, window = 10, workers =4)
#    model = gensim.models.Word2Vec(min_count=10, workers = 4)
#    model.build_vocab(sentences_)
#    model.train(sentences_, total_examples=model.corpus_count, epochs=model.iter)
#    
#    print('....saving')
#    model.init_sims(replace=True)
#    model.wv.save_word2vec_format(modelName, binary=True)



#def modelYear(year):
#    '''
#    Build a model for just one year
#    '''
#    modelName = modelFolder + '/%d_%d.w2v'%(year,year+yearsInModel)
#    sentences = getSentencesForYear(year)
#    model = gensim.models.Word2Vec(min_count=10, size = 200, iter=10, alpha=0.025, window=10, workers =4)
#    model.build_vocab(sentences)
#    model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
#    model.init_sims(replace=True)
#    #model.wv.save_word2vec_format(modelName, binary=True)
#    return model
#
##modelYear(1971)

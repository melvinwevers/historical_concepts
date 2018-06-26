#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 15:44:22 2017

@author: wevers
"""

import os, io, glob
import numpy as np
import pandas as pd
import re
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import string
from collections import Counter
from nltk.tag import pos_tag
from gensim import corpora, models

path = 'data2'
stoplist = stopwords.words('dutch')

#allFiles = glob.glob(path + "/clean*.tsv")
#frame = pd.DataFrame() 
#list_ = []
#for file_ in allFiles:
#    frame = pd.read_csv(file_, sep='\t', encoding='utf-8')
#    list_.append(frame)
#df = pd.concat(list_)

#word count function
def freq_dist(data, n_words):
    ngram_vectorizer = CountVectorizer(analyzer='word', tokenizer=word_tokenize, ngram_range=(1, 1), min_df=0.01, max_df=0.90)
    X = ngram_vectorizer.fit_transform(data)
    vocab = list(ngram_vectorizer.get_feature_names())
    counts = X.sum(axis=0).A1
    freq_distribution = Counter(dict(zip(vocab, counts)))
    return freq_distribution.most_common(n_words)

#print(freq_dist(df['ocr'], 100))

def tf_idf(data):
    tf_idfvectorizer = TfidfVectorizer(norm='l1', sublinear_tf=True, analyzer='word', lowercase=True, min_df = 0.01, max_df=0.90)
    tfidf_matrix =  tf_idfvectorizer.fit_transform(data)
    scores = zip(tf_idfvectorizer.get_feature_names(), np.asarray(tfidf_matrix.sum(axis=0)).ravel())
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    feature_list = []
    for item in sorted_scores:
        feature_list.append("{0:50} Score: {1}".format(item[0], item[1]))
    return feature_list



dictionary = corpora.Dictionary


docs = df2["ocr"][0:100].apply(word_tokenize)
docs2 = df["tokenized_articles"].apply(word_tokenize)

dictionary = corpora.Dictionary(docs)
bow = [dictionary.doc2bow(doc) for doc in docs]
k = 10
lda = models.LdaModel(bow, id2word = dictionary, num_topics = k, random_state = 1234, passes=5, chunksize=10000, update_every=5)

for i in range(k):
    print('Topic', i)
    print([t[0] for t in lda.show_topic(i,10)])
    print('-----')


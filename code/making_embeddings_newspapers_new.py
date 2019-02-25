#!/usr/bin/env python
'''
Generate embeddings with a time shifting window.

Usage:
  train_models.py --y0=<y0> --yN=<yN> --nYears=<years> --title=<title> --outDir=<dir> [--step=<years>]

Options:
  --y0 <y0>         First year in the generated models
  --yN <yN>         Last year in the generated models
  --nYears <years>  Number of years per model
  --title <title>   Name of the input data 
  --outDir <dir>    Directory where models will be writen to
  --step <years>    Step between start year of generated models [default: 1]
'''


from docopt import docopt
import gensim
from gensim.models.word2vec import Word2Vec
from gensim.models.word2vec import LineSentence
import itertools
import os
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


num_features = 300    # Word vector dimensionality
min_word_count = 5   # Minimum word count
context = 10         # Context window size
downsampling = 10e-5  # Downsample setting for frequent words
num_workers = 10
hierarchical_softmax = 1
skip_gram = 1
negative_sampling_num_words = 0


class Sentences():
    def __init__(self, data_path):
        self.data_path = data_path

    def __iter__(self):
        return iter_file(self.data_path)


class TimestampedSentences():
    def __init__(self, start_year, end_year, data_path):
        self.start_year = start_year
        self.end_year = end_year
        self.data_path = data_path

    def __iter__(self):
        return itertools.chain.from_iterable(iter_load_sentences(self.start_year, self.end_year, self.data_path))


def iter_file(path):
    '''
    load file and generate sentences
    '''
    with open(path, 'r') as f:
        for sentence in f:
            yield sentence.split()


def iter_load_sentences(start_year, end_year, data_path):
    '''
    load sentences in a date range
    '''
    for year in range(start_year, end_year + 1):
            year_path = os.path.join(data_path, 'vk_' + str(year) + '.txt')
            yield iter_file(year_path)

def train_embeddings(sentences, num_features=300,
                     min_word_count=5, num_workers=10, context=10, downsampling=1e-3, sg=1,
                     hierarchical_softmax=0, negative_sampling_num_words=5):
    model = Word2Vec(workers=num_workers,
                     size=num_features, min_count=min_word_count,
                     window=context, sample=downsampling,
                     sg=sg, hs=hierarchical_softmax, negative=negative_sampling_num_words)
    '''
    training wem and generate bigrams
    '''
    #bigram_transformer = gensim.models.Phrases(sentences, min_count=100)
    #bigram = gensim.models.phrases.Phraser(bigram_transformer)
    #corpus = list(bigram[sentences])
    model.build_vocab(sentences)
    model.train(sentences,
                total_examples=model.corpus_count,
                #epochs=model.iter
                epochs=5)
    return model.wv


def save_embeddings(embeddings, path, file):
    model_ = "{0}{1}.w2v".format(path, file)
    vocab_ = model_.replace('w2v', 'vocab.w2v')
    try:
        #embeddings.save(save_path)
        embeddings.save_word2vec_format(model_, fvocab=vocab_, binary=True)
    except FileNotFoundError:
        print(path)
        os.makedirs(path)
        embeddings.save_word2vec_format(model_, fvocab=vocab_, binary=True)


def train_models(y0, yN, yearsInModel, title, stepYears, modelFolder):
    '''train model and specify beginning and end and size of model
    '''
    for year in range(y0, yN+1, stepYears):
        startY = year
        endY = year + yearsInModel-1
        modelName = '%d_%d' % (startY, endY)
        print('Building Model: ', modelName)

        periods = [(modelName, TimestampedSentences(
            startY, endY, '../data/{}'.format(title)))]
        
        for identifier, sentences in periods:
        #for sentences in TimestampedSentences(startY, endY, '../code/articles'):
            embeddings = train_embeddings(sentences, num_features=num_features, min_word_count=min_word_count, num_workers=num_workers,
                                          context=context, downsampling=downsampling, sg=skip_gram,
                                          hierarchical_softmax=hierarchical_softmax,
                                          negative_sampling_num_words=negative_sampling_num_words)
            save_path = os.path.join(modelFolder + title + '/' + str(stepYears))
            print(save_path)
            save_embeddings(embeddings, save_path, '/{}'.format(identifier))


if __name__ == '__main__':
    args = docopt(__doc__)
    yearsInModel = int(args['--nYears'])
    stepYears = int(args['--step'])
    title = args['--title']
    outDir = args['--outDir']
    y0 = int(args['--y0'])
    yN = int(args['--yN'])

    train_models(y0, yN, yearsInModel, title, stepYears, outDir)

#!/usr/bin/env python
'''
Preprocess newspaper data for training embeddings

Usage: preprocessing.py --title=<newspaper>

Options:
 --title <newspaper>    Title of the newspaper

'''


import glob
import pandas as pd
from docopt import docopt
import re
import itertools
import unidecode
import nltk
from nltk.corpus import stopwords
from gensim.parsing.preprocessing import preprocess_string
from gensim.parsing.preprocessing import *
import gensim
import gensim.downloader as api
from gensim.utils import save_as_line_sentence
from gensim.models.word2vec import Word2Vec

StopWords = set(stopwords.words('dutch'))

def article_to_sentences(text):
    '''
    split article into sentences
    '''
    sent_tokenizer = nltk.punkt.PunktSentenceTokenizer()
    sentences = clean_sentences(sent_tokenizer.tokenize(text))
    return sentences


def clean_sentences(sentences):
    '''
    clean sentences by removing punctuation, accents,
    removing words shorter than 3 characters and stopwords
    '''
    cleanSentences = []
    for sentence in sentences:
        sentence = re.sub("[^a-zA-Z]", " ", sentence)
        sentence = ' '.join(word.lower() for word in sentence.split() if len(word) >= 2 and word not in StopWords)
        if len(sentence) > 0:
            cleanSentences.append(sentence)
    return cleanSentences

def process_corpus(docs):
    CUSTOM_FILTERS = [lambda x: x.lower(), strip_numeric, strip_multiple_whitespaces, strip_punctuation, strip_short]
    for doc in docs:

        yield preprocess_string(doc, CUSTOM_FILTERS)

def load_file(title):
    path = '../../newspapers/{}'.format(title)
    allFiles = glob.glob(path + "/articles/*.tsv")

    for f in allFiles:
        df = pd.read_csv(f, delimiter='\t', header=None)
        df.columns = ['date', 'page', 'size', 'min_x', 'min_y',
                  'max_x', 'max_y', 'w', 'h', 'image_url', 'ocr_url', 'ocr']
        df = df.dropna(subset=['ocr'])  # remove lines with empty ocr field

        df = df[~df['date'].str.contains('date')]  # remove duplicate header rows
        # remove files that contain error msg
        excludes = ['objecttype', 'file directory not found']
        df = df[~df['ocr'].astype(str).str.contains('|'.join(excludes))]
        df['date'] = pd.to_datetime(df['date'])
        year = df['date'].dt.year[1]
        print('making sentences: {}'.format(year))
        df['ocr'] = df['ocr'].apply(lambda x: unidecode.unidecode(x))
        docs = df['ocr'].values
        CORPUS_FILE = ('{}_{}.txt'.format(title, year))

        save_as_line_sentence(process_corpus(docs), CORPUS_FILE)

        # df['ocr'] = df['ocr'].apply(lambda x: unidecode.unidecode(x))
        # docs = df['ocr'].apply(article_to_sentences)
        # with open(out_path +'/{}_{}.txt'.format(title, year), 'w') as output:
        #     for doc in docs:
        #         for sentence in doc:
        #             output.write("\n%s" % sentence)
       

if __name__ == '__main__':
    args = docopt(__doc__)

    title = args['--title']
    #out_path = args['--outDir']

    load_file(title)

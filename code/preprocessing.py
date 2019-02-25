#!/usr/bin/env python
'''
Preprocess newspaper data for training embeddings

Usage: preprocessing.py --title=<newspaper> --outDir=<dir>

Options:
 --title <newspaper>    Title of the newspaper
 --outDir <dir>         Path for output 

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
from tqdm import tqdm
import gensim.downloader as api
from gensim import utils
from gensim.utils import save_as_line_sentence
from gensim.models.word2vec import Word2Vec

StopWords = frozenset(stopwords.words('dutch'))

def process_corpus(docs):

    with open('../dictionary.dict') as f:
        dictionary_nl = f.readlines()
    dictionary_nl = [x.strip() for x in dictionary_nl]
    dictionary_nl = frozenset(dictionary_nl)

    def dict_check(s):
        s = utils.to_unicode(s)
        return " ".join(w.lower() for w in s.split() if w.lower() in dictionary_nl and w.lower() not in StopWords)

    CUSTOM_FILTERS = [strip_numeric, strip_multiple_whitespaces, strip_punctuation, strip_short, lambda x: dict_check(x)]
    for doc in docs:
        yield preprocess_string(doc, CUSTOM_FILTERS)

def load_file(title, out_path):
    path = '../../../datasets/newspapers_clean/{}'.format(title)
    print(path)
    allFiles = glob.glob(path + "/articles/*.tsv")
    print(allFiles)

    for f in allFiles:
        df = pd.read_csv(f, delimiter='\t', parse_dates=True)
        #df.columns = ['date', 'page', 'size', 'min_x', 'min_y',
        #          'max_x', 'max_y', 'w', 'h', 'image_url', 'ocr_url', 'ocr']
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
        CORPUS_FILE = (out_path + '/{}_{}.txt'.format(title, year))

        save_as_line_sentence(process_corpus(docs), CORPUS_FILE)
        # with open('../dictionary.dict') as f:
        #     dictionary_nl = f.readlines()
        # dictionary_nl = [x.strip() for x in dictionary_nl]

        # with open('{}_{}_clean.txt'.format(title, year), 'w') as outfile:
        #     with open(CORPUS_FILE) as infile:
        #         for line in tqdm(infile):
        #             line = ' '.join(word for word in line.split() if word in dictionary_nl)
        #             outfile.write("\n%s" % line)


        # df['ocr'] = df['ocr'].apply(lambda x: unidecode.unidecode(x))
        # docs = df['ocr'].apply(article_to_sentences)
        # with open(out_path +'/{}_{}.txt'.format(title, year), 'w') as output:
        #     for doc in docs:
        #         for sentence in doc:
        #             output.write("\n%s" % sentence)
       

if __name__ == '__main__':
    args = docopt(__doc__)

    title = args['--title']
    out_path = args['--outDir']

    load_file(title, out_path)

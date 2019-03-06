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
import unidecode
from nltk.corpus import stopwords
from gensim.parsing.preprocessing import preprocess_string
from gensim.parsing.preprocessing import *
from gensim import utils
from gensim.utils import save_as_line_sentence

StopWords = frozenset(stopwords.words('dutch'))


def process_corpus(docs):

    with open('../dictionary.dict') as f:
        dictionary_nl = f.readlines()
    dictionary_nl = [x.strip() for x in dictionary_nl]
    dictionary_nl = frozenset(dictionary_nl)

    def dict_check(s):
        s = utils.to_unicode(s)
        return " ".join(w.lower() for w in s.split() if w.lower()
                        in dictionary_nl and w.lower() not in StopWords)

    CUSTOM_FILTERS = [strip_numeric,
                      strip_multiple_whitespaces,
                      strip_punctuation,
                      strip_short,
                      lambda x: dict_check(x)]
    for doc in docs:
        yield preprocess_string(doc, CUSTOM_FILTERS)


def load_file(title, out_path):
    path = '../../../datasets/newspapers_clean/{}'.format(title)
    print(path)
    allFiles = glob.glob(path + "/articles/*.tsv")
    print(allFiles)

    for f in allFiles:
        df = pd.read_csv(f, delimiter='\t', parse_dates=True)
        df = df.dropna(subset=['ocr'])  # remove lines with empty ocr field

        # remove duplicate header rows
        df = df[~df['date'].str.contains('date')]
        # remove files that contain error msg
        excludes = ['objecttype', 'file directory not found']
        df = df[~df['ocr'].astype(str).str.contains('|'.join(excludes))]

        df['date'] = pd.to_datetime(df['date'])
        year = df['date'].dt.year[0]

        print('making sentences: {}'.format(year))
        df['ocr'] = df['ocr'].apply(lambda x: unidecode.unidecode(x))
        docs = df['ocr'].values
        CORPUS_FILE = (out_path + '/{}_{}.txt'.format(title, year))

        save_as_line_sentence(process_corpus(docs), CORPUS_FILE)


if __name__ == '__main__':
    args = docopt(__doc__)

    title = args['--title']
    out_path = args['--outDir']

    load_file(title, out_path)

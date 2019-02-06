#!/usr/bin/env python
'''
Preprocess newspaper data for training embeddings

Usage: preprocess.py --title=<newspaper> --outDir=<dir>

Options:
 --title <newspaper>    Title of the newspaper
 --outDir <dir>         Directory where preprocessed files are saved

'''


import pandas as pd
import glob
import os
import re
from nltk.corpus import stopwords
import unidecode
from tqdm import tqdm
from sys import argv
import numpy as np
from docopt import docopt
from gensim.models.phrases import Phrases, Phraser

stop_words = set(stopwords.words('dutch'))

# script, title = argv

# out_path = '../data/{}/'.format(title)

def load_newspapers(title, out_path):
    regex_pat = re.compile(r'[^a-zA-Z\s]', flags=re.IGNORECASE)
    path = '../../newspapers/{}'.format(title)
    print(path)
    allFiles = glob.glob(path + '/*.csv')
    bigFile = []
    for f in tqdm(allFiles):
        print(os.path.basename(f))
        filename_ = os.path.basename(f)
        year_ = filename_[13:17]
        df = pd.read_csv(f, delimiter='\t')
       
        df['text'] = df['text'].astype(str)
       
        #df['perc_digits'] = df['text'].apply(lambda x: digit_perc(x))
        #df = df[df['perc_digits'] <= 0.5]
        
        df['text'] = df['text'].apply(lambda x: unidecode.unidecode(x)) #I could also use Gensim preprocess for this now but now it's the same as dictionary
        df['text'] = df['text'].str.replace(regex_pat, '')
        df['text'] = df['text'].str.findall(r'\w{3,18}').str.join(' ') #Only select words between 3 and 17 characters
        
        #df['text'] = df['text'].apply(lambda x: make_bigrams(x))
        
        df['text'] = df['text'].str.lower()
        
        
        #df = df[df['len'].between(250, 5000, inclusive=True)]
        df['text'] = df['text'].apply(lambda x: remove_stopwords(x))
        df['len'] = df['text'].str.split().apply(len)
        directory = out_path + str(year_)   
        
        if not os.path.exists(directory):
            os.makedirs(directory)

        cols = ['text', 'title']
        df2 = df[cols].copy()
        df2.rename(columns={'text': 'Content', 'title': 'Title'}, inplace=True)
        df2.to_csv(directory + '/' + filename_[:-3] + 'csv', sep=',')
        bigFile.append(df)
    bigFile = pd.concat(bigFile)
    bigFile.to_pickle(directory + '/{}.pkl'.format(title))


def make_bigrams(text):
    '''
    Construct bigrams from word pairs occuring more than 50 times
    '''

    words = text.split(" ")
    bigram_transformer = Phrases(words, min_count=50)
    bigram = Phraser(bigram_transformer)
    bigrams = list(bigram[words])
    return ' '.join(bigram for bigram in bigrams)


def remove_stopwords(texts):
    '''
    remove accents and increase max length of words
    Dutch has longer words than English
    '''
    return ''.join(word for word in texts if word not in stop_words)

def digit_perc(x):
    '''
    Calculate percentage of digits per character in text.
    Too many digits refers to sports results, tv guides, or shipping reports.
    '''
    return sum(c.isdigit() for c in str(x)) / len(str(x))

if __name__ == "__main__":
    args = docopt(__doc__)

    title  = args['--title']
    out_path = args['--outDir'] 
    
    load_newspapers(title, out_path)
    

        

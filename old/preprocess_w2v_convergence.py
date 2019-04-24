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
import numpy as np
from docopt import docopt
from gensim.models.phrases import Phrases, Phraser

stop_words = set(stopwords.words('dutch'))


def load_newspapers(title, out_path):
    regex_pat = re.compile(r'[^a-zA-Z\s]', flags=re.IGNORECASE)
    path = '../../../datasets/newspapers_clean/{}'.format(title)
    print(path)
    allFiles = glob.glob(path + '/articles/*.*sv')

    for f in tqdm(allFiles):
        print(os.path.basename(f))
        filename_ = os.path.basename(f)
        year_ = filename_[12:16]
        df = pd.read_csv(f, delimiter='\t')
       
        # remove double headers
        df = df[~df['date'].astype(str).str.contains('date')]  
        df = df[~df['ocr'].astype(str).str.contains('objecttype')]  
        df['ocr'] = df['ocr'].astype(str)
        
        #df['perc_digits'] = df['ocr'].apply(lambda x: digit_perc(x))
        #df = df[df['perc_digits'] <= 0.5]
        
        #I could also use Gensim preprocess for this 
        df['ocr'] = df['ocr'].apply(lambda x: unidecode.unidecode(x)) 
        df['ocr'] = df['ocr'].str.replace(regex_pat, '')
        #Only select words between 3 and 17 characters
        df['ocr'] = df['ocr'].str.findall(r'\w{3,18}').str.join(' ') 
        
        #df['ocr'] = df['ocr'].apply(lambda x: make_bigrams(x))
        
        df['ocr'] = df['ocr'].str.lower()
        
        df['len'] = df['ocr'].str.split().apply(len)
        #df = df[df['len'].between(250, 5000, inclusive=True)]
        df['ocr'] = df['ocr'].apply(lambda x: remove_stopwords(x))
        directory = out_path + str(year_)   
        
        if not os.path.exists(directory):
            os.makedirs(directory)

        df['Content'] = df['ocr']
        df['Title'] = title

        cols = ['Title', 'Content']
        df = df[cols]
        df.to_csv(directory + '/' + filename_[:-3] + 'csv', sep=',')



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
    

        
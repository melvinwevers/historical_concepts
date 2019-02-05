import pandas as pd
import glob
import os
import re
from nltk.corpus import stopwords
import unidecode
from tqdm import tqdm
from sys import argv
import numpy as np
from gensim.models import Phrases

stop_words = set(stopwords.words('dutch'))

script, title = argv

out_path = '../data/{}/'.format(title)

def load_newspapers(title, concat=False, sample=False):
    regex_pat = re.compile(r'[^a-zA-Z\s]', flags=re.IGNORECASE)
    path = '../../newspapers/{}'.format(title)
    print(path)
    allFiles = glob.glob(path + '/articles/*.tsv')

    for f in tqdm(allFiles):
        print(os.path.basename(f))
        filename_ = os.path.basename(f)
        year_ = filename_[12:16]
        df = pd.read_csv(f, delimiter='\t')
        df = df[~df['date'].str.contains('date')]  # remove double headers
        df = df[~df['ocr'].str.contains(
            'objecttype')]  # remove double headers
        df['ocr'] = df['ocr'].astype(str)
        #df['perc_digits'] = df['ocr'].apply(lambda x: digit_perc(x))
        #df = df[df['perc_digits'] <= 0.5]
        df['ocr'] = df['ocr'].apply(lambda x: unidecode.unidecode(x)) #I could also use Gensim preprocess for this now but now it's the same as dictionary
        df['ocr'] = df['ocr'].str.replace(regex_pat, '')
        df['ocr'] = df['ocr'].str.findall(r'\w{3,18}').str.join(' ') #Only select words between 3 and 17 characters
        df['ocr'] = df['ocr'].apply(lambda x: make_bigrams(x))
        df['ocr'] = df['ocr'].str.lower()
        df['len'] = df['ocr'].str.split().apply(len)
        #df = df[df['len'].between(250, 5000, inclusive=True)]
        df['ocr'] = df['ocr'].apply(lambda x: remove_stopwords(x))
        directory = out_path + str(year_)   
        if not os.path.exists(directory):
            os.makedirs(directory)

        df['Content'] = df['ocr']
        #docs = df['ocr'].values

        df['Title'] = title

        cols = ['Title', 'Content']
        df = df[cols]
        df.to_csv(directory + '/' + filename_[:-3] + 'csv', sep=',')


        #np.savetxt(directory + '/' + filename_[:-3] + "csv", docs, delimiter=",", fmt='%s')


def make_bigrams(text):
    words = text.split(" ")
    bigram_transformer = gensim.models.Phrases(words, min_count=25)
    bigram = gensim.models.phrases.Phraser(bigram_transformer)
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

    return df

if __name__ == "__main__":
    
    df = load_newspapers(title)
    print('preprocessed data')
    

        

import glob
import pandas as pd
import re
import logging
from nltk.tokenize import sent_tokenize
from nltk.tokenize.treebank import TreebankWordTokenizer

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)
tokenizer = TreebankWordTokenizer()


def cleaning(text):
    text = re.sub("[^a-zA-Z]", " ", text)
    return(text)


def article_to_sentences(text):
    all_txt = []
    sentences = sent_tokenize(cleaning(text.lower().strip()))
    sentences = [tokenizer.tokenize(sent) for sent in sentences]
    all_txt += sentences
    return all_txt


def pre_process():
    path = '../data/newspapers/ah'
    allFiles = glob.glob(path + "/*.tsv")

    df = pd.concat((pd.read_csv(f, delimiter='\t',
                                header=None) for f in allFiles))
    df.columns = ['date', 'page', 'size', 'min_x', 'min_y',
                  'max_x', 'max_y', 'w', 'h', 'image_url', 'ocr_url', 'ocr']
    df = df.dropna(subset=['ocr'])  # remove lines with empty ocr field

    df = df[~df['date'].str.contains('date')]  # remove duplicate header rows
    # remove files that contain error msg
    excludes = ['objecttype', 'file directory not found']
    df = df[~df['ocr'].astype(str).str.contains('|'.join(excludes))]
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    for name, group in df.groupby('year'):
        print('making sentences: {}'.format(name))
        sentences = group['ocr'].apply(article_to_sentences)
        sentences = [item for sublist in sentences for item in sublist]
        output_ = open(str(name) + '.txt', 'w')
        for sentence in sentences:
            sentence = ' '.join(word for word in sentence)
            output_.write("%s\n" % sentence)
        output_.close()

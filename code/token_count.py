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
    text = [word.lower() for word in text.split() if len(word) >= 3]
    #text = ' '.join(text.lower().split())
    return(len(text))


# def article_to_sentences(text):
#     all_txt = []
#     sentences = sent_tokenize(cleaning(text.lower().strip()))
#     sentences = [tokenizer.tokenize(sent) for sent in sentences]  
#     all_txt += sentences
#     return all_txt


def pre_process():
    path = '../data/newspapers/vk/'
    title_ = 'vk'
    allFiles = glob.glob(path + "/*.tsv")

    for f in allFiles:
        print(f)
        df = pd.read_csv(f, delimiter='\t', header=None)
        df.columns = ['date', 'page', 'size', 'min_x', 'min_y',
                      'max_x', 'max_y', 'w', 'h', 'image_url', 'ocr_url', 'ocr']
        df = df.dropna(subset=['ocr'])  # remove lines with empty ocr field

        # remove duplicate header rows
        df = df[~df['date'].str.contains('date')]
        # remove files that contain error msg
        excludes = ['objecttype', 'file directory not found']
        df = df[~df['ocr'].astype(str).str.contains('|'.join(excludes))]
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['len'] = df['ocr'].apply(cleaning)
        df.to_csv(f)

if __name__ == '__main__':
    pre_process()


    df.to_pickle('df2.pkl')


if __name__ == '__main__':
    pre_process()

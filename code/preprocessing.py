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
    text = ' '.join(word.lower() for word in text.split() if len(word) >= 3)
    #text = ' '.join(text.lower().split())
    return(text)


# def article_to_sentences(text):
#     all_txt = []
#     sentences = sent_tokenize(cleaning(text.lower().strip()))
#     sentences = [tokenizer.tokenize(sent) for sent in sentences]  
#     all_txt += sentences
#     return all_txt

def article_to_sentences(text):
	#raw_sentences = tokenizer.tokenize(text.strip())
	raw_sentences = text.split('.')
	sentences = []
	for raw_sentence in raw_sentences:
		if len(raw_sentence) > 0:
			sentences.append(cleaning(raw_sentence))
	return sentences


def pre_process():
    path = '../data/newspapers/vk/'
    title_ = 'vk'
    allFiles = glob.glob(path + "/*.tsv")

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
        df['year'] = df['date'].dt.year
        for year, group in df.groupby('year'):
            print('making sentences: {}'.format(year))
            sentences = group['ocr'].apply(article_to_sentences)
            group['sentences'] = sentences
            #sentences = [item for sublist in sentences for item in sublist]
            output_ = open('{}_{}.txt'.format(title_, year), 'w')
            for sentence in sentences:
                #sentence = ''.join(word for word in sentence)
                output_.write("\n%s" % sentence)
            output_.close()

if __name__ == '__main__':
    pre_process()

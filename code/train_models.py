import glob
import pandas as pd
import re
import nltk
import gensim
import logging
import random
#from gensim.models import FastText
#from gensim.similarities.index import AnnoyIndexer
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from gensim.models.word2vec import LineSentence
import glob
import os
from gensim.models import KeyedVectors

#from functools32 import lru_cache

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

path = '../newspapers2018/vk/articles'
#path ='test'
path2 = 'sentences'
#path2 = 'test2'
allFiles = glob.glob(path + "/*.tsv")
#print(allFiles)
tokenizer = nltk.data.load('tokenizers/punkt/dutch.pickle')

def cleaning(text):
	text = re.sub("[^a-zA-Z]"," ", text)
	words = text.lower().split()
	return(words)

def article_to_sentences(text):
	#raw_sentences = tokenizer.tokenize(text.strip())
	raw_sentences = text.split('.')
	sentences = []
	for raw_sentence in raw_sentences:
		if len(raw_sentence) > 0:
			sentences.append(cleaning(raw_sentence))
	return sentences

def getSentencesForYear(year):
	corpus = []
	for file_ in allFiles:
		filename = re.sub(path + '/', '', file_)
		filename = filename[12:]
		if filename.startswith(str(year)):
			df = pd.read_csv(file_, sep='\t', encoding='utf-8', header=None)
			df.columns = ['date', 'page', 'size', 'min_x', 'min_y', 'max_x', 'max_y', 'w', 'h', 'image_url', 'ocr_url', 'ocr']
			df = df[df['ocr'].notnull()] #remove empty rows
			sentences = df['ocr'].apply(article_to_sentences)
			sentences.to_csv(path2 + '/'+ 'sentences' + filename, index=False, sep='\t', encoding='utf-8')
			sentences = [item for sublist in sentences for item in sublist]
			bigram_transformer = gensim.models.Phrases(sentences)
			bigram = gensim.models.phrases.Phraser(bigram_transformer)
			corpus = list(bigram[sentences])
	return corpus
		

def getSentencesInRange(startY, endY):
	return [s for year in range(startY, endY) for s in getSentencesForYear(year)]    


def train_models():
	yearsInModel = 1    
	stepYears = 1
	modelFolder = 'tempModels_vk'

	y0 = 1945
	yN = 1995

	for year in range(y0, yN-yearsInModel+1, stepYears):
		startY = year
		endY = year + yearsInModel
		modelName = modelFolder + '/%d_%d.w2v'%(year,year+yearsInModel)
		print('Building Model: ', modelName)
		
		corpus = getSentencesInRange(startY, endY)

		model = gensim.models.Word2Vec(min_count=10, size = 200, iter=50, alpha=0.025, window = 10, workers = 40, sg=1)
		model.build_vocab(corpus)
		model.train(corpus , total_examples=model.corpus_count, epochs=model.iter)
		print('....saving')
		model.init_sims(replace=True)
		model.wv.save_word2vec_format(modelName, binary=True)

		
if __name__ == '__main__':
	train_models()



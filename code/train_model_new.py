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
import spacy
from gensim.models import KeyedVectors

#from functools32 import lru_cache

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

nlp = spacy.blank('nl')
nlp.add_pipe(nlp.create_pipe('sentencizer'))

print('Pipeline: ', nlp.pipe_names)

path = '../newspapers2018/vk/articles'
#path ='test'
path2 = 'sentences_new'
#path2 = 'test2'
allFiles = glob.glob(path + "/*.tsv")
#print(allFiles)
tokenizer = nltk.data.load('tokenizers/punkt/dutch.pickle')

# def cleaning(text):
# 	text = re.sub("[^a-zA-Z]"," ", text)
# 	words = text.lower().split()
# 	return(words)

# def article_to_sentences(text):
# 	#raw_sentences = tokenizer.tokenize(text.strip())
# 	raw_sentences = text.split('.')
# 	sentences = []
# 	for raw_sentence in raw_sentences:
# 		if len(raw_sentence) > 0:
# 			sentences.append(cleaning(raw_sentence))
# 	return sentences

def is_noise(token, remove_stopwords = True, remove_punctuation = True):
    token_flags = [token.is_stop, token.is_punct]
    do_flags = [remove_stopwords, remove_punctuation]
    flags = [token and do for token, do in zip(token_flags, do_flags) ]
    return any(flags)

def normalize(token, do_lemmatize = True, do_lower = True ):
    if do_lemmatize and do_lower:
        return token.lemma_.lower()
    if do_lemmatize:
        return token.lemma_
    if do_lower:
        return token.lower_


def extract_sentences(texts, do_lower = True, do_lemmatize = True, remove_stopwords = True, remove_punctuation = True):
	for doc in nlp.pipe(texts, batch_size = 10000, n_threads = 10):
		yield [[normalize(token, do_lower, do_lemmatize) for token in sent 
		        if not is_noise(token, remove_stopwords, remove_punctuation)]
				for sent in doc.sents]


def getSentencesForYear(year):
	corpus = []
	for file_ in allFiles:
		filename = re.sub(path + '/', '', file_)
		filename = filename[12:]		
		if filename.startswith(str(year)):
			print(filename)
			df = pd.read_csv(file_, sep='\t', encoding='utf-8', header=None)
			df = df.sample(frac=0.0001, replace=False)
			df.columns = ['date', 'page', 'size', 'min_x', 'min_y', 'max_x', 'max_y', 'w', 'h', 'image_url', 'ocr_url', 'ocr']
			df = df[df['ocr'].notnull()] #remove empty rows

			df['ocr'].to_csv('sentences.txt', index=False)

			with open('sentences.txt', 'r') as in_file:
				articles = (article for article in in_file)
				for sentences in extract_sentences(articles, do_lemmatize=False):
					print(*[' '.join(sentence) for sentence in sentences if sentence],
                                            file='test.txt',
                                            sep=os.linesep)







			
			#sentences = df['ocr'].apply(extract_sentences, do_lemmatize = False)
			#print(*[ [item for sublist in sentences for item in sublist]
			#sentences.to_csv(path2 + '/'+ 'sentences_new' + filename, index=False, sep='\t', encoding='utf-8')
			
			#bigram_transformer = gensim.models.Phrases(sentences)
			#bigram = gensim.models.phrases.Phraser(bigram_transformer)
			#corpus = list(bigram[sentences])
	return corpus
		

def getSentencesInRange(startY, endY):
	return [s for year in range(startY, endY) for s in getSentencesForYear(year)]    


def train_models():
	yearsInModel = 1    
	stepYears = 1
	modelFolder = 'tempModels_vk'

	y0 = 1960
	yN = 1961

	for year in range(y0, yN-yearsInModel+1, stepYears):
		startY = year
		endY = year + yearsInModel
		modelName = modelFolder + '/%d_%d.w2v'%(year,year+yearsInModel)
		print('Building Model: ', modelName)
		
		corpus = getSentencesInRange(startY, endY)

		model = gensim.models.Word2Vec(min_count=5, size = 300, window = 10, workers = 40, sg=1, sample = 1e-3, hs = 0, )
		model.build_vocab(corpus)
		model.train(corpus , total_examples=model.corpus_count, epochs=model.iter)
		print('....saving')
		model.init_sims(replace=True)
		model.wv.save_word2vec_format(modelName, binary=True)

		
if __name__ == '__main__':
	#train_models()
	getSentencesForYear(1960)



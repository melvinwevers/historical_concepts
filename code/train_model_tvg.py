import pickle
import gensim
import logging


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

with open ('outfile', 'rb') as fp:
    corpus = pickle.load(fp)


def prepare_model(data):
	corpus = [item for sublist in data for item in sublist]
	bigram_transformer = gensim.models.Phrases(corpus, min_count=10)
	bigram = gensim.models.phrases.Phraser(bigram_transformer)
	corpus_ = list(bigram[corpus])
	return corpus_

def train_model(data):
	corpus_ = prepare_model(data)
	model = gensim.models.Word2Vec(min_count=10, size = 200, iter=20, alpha=0.025, window = 10, workers =40, sg=1)
	model.build_vocab(corpus_)
	model.train(corpus_, total_examples=model.corpus_count, epochs=model.iter)
	model.init_sims(replace=True)
	model.wv.save_word2vec_format('tvg', binary=True)

if __name__ == '__main__':
	train_model(corpus)
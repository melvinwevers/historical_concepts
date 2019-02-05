import gensim
from gensim.models.word2vec import Word2Vec
import itertools
import os
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


num_features = 300    # Word vector dimensionality
min_word_count = 5   # Minimum word count
context = 10         # Context window size
downsampling = 10e-5  # Downsample setting for frequent words
num_workers = 30
hierarchical_softmax = 1
skip_gram = 1
negative_sampling_num_words = 0


class Sentences():
    def __init__(self, data_path):
        self.data_path = data_path

    def __iter__(self):
        return iter_file(self.data_path)


class TimestampedSentences():
    def __init__(self, start_year, end_year, data_path):
        self.start_year = start_year
        self.end_year = end_year
        self.data_path = data_path

    def __iter__(self):
        return itertools.chain.from_iterable(iter_load_sentences(self.start_year, self.end_year, self.data_path))


def iter_file(path):
    '''
    load file and generate sentences
    '''
    with open(path, 'r') as f:
        for sentence in f:
            if sentence.split():
                yield sentence.lower().split()


def iter_load_sentences(start_year, end_year, data_path):
    '''
    load sentences in a date range
    '''
    for year in range(start_year, end_year + 1):
            year_path = os.path.join(data_path, 'vk_' + str(year) + '.txt')
            yield iter_file(year_path)

def train_embeddings(sentences, num_features=300,
                     min_word_count=5, num_workers=5, context=10, downsampling=1e-3, sg=1,
                     hierarchical_softmax=0, negative_sampling_num_words=5):
    model = Word2Vec(workers=num_workers,
                     size=num_features, min_count=min_word_count,
                     window=context, sample=downsampling,
                     sg=sg, hs=hierarchical_softmax, negative=negative_sampling_num_words)
    '''
    training wem and generate bigrams
    '''
    bigram_transformer = gensim.models.Phrases(sentences, min_count=50)
    bigram = gensim.models.phrases.Phraser(bigram_transformer)
    corpus = list(bigram[sentences])
    model.build_vocab(corpus)
    model.train(corpus,
                total_examples=model.corpus_count,
                epochs=model.iter)
    return model.wv


def save_embeddings(embeddings, path, file):
    save_path = "{0}{1}".format(path, file)
    try:
        embeddings.save(save_path)
    except FileNotFoundError:
        os.mkdir(path)
        embeddings.save(save_path)


def train_models(y0, yN, yearsInModel=10, stepYears=10):
    '''train model and specify beginning and end and size of model
    '''
    #yearsInModel = 20    
    #stepYears = 20
    for year in range(y0, yN+1, stepYears):
        startY = year
        endY = year + yearsInModel-1
        modelName = '%d_%d' % (startY, endY)
        print('Building Model: ', modelName)

        periods = [(modelName, TimestampedSentences(
            startY, endY, 'sentences_vk'))]
        
        for identifier, sentences in periods:
        #for sentences in TimestampedSentences(startY, endY, '../code/articles'):
            embeddings = train_embeddings(sentences, num_features=num_features, min_word_count=min_word_count, num_workers=num_workers,
                                          context=context, downsampling=downsampling, sg=skip_gram,
                                          hierarchical_softmax=hierarchical_softmax,
                                          negative_sampling_num_words=negative_sampling_num_words)
            save_embeddings(embeddings, 'embeddings/year',"{0}".format(identifier))


if __name__ == '__main__':
	train_models(1960, 1960, 1, 1)

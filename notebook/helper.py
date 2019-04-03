import gensim
from gensim.models import KeyedVectors
import numpy as np

def cossim(v1, v2, signed = True):
    '''
    calculate cossimilarity between two vectors
    '''
    c = np.dot(v1, v2)/np.linalg.norm(v1)/np.linalg.norm(v2)
    if not signed:
        return abs(c)
    return c

def quantile_function(df, lower, upper, col = 'value'):
    '''
    return data that falls between lower and upper quantile values
    '''
    lower, upper = df.quantile([lower, upper])[col]
    return df.query('{lower}<counts<{upper}'.format(lower, upper))

def calc_distance_between_vectors(vec1, vec2, distype = 'norm'):
    if distype is 'norm':
        return np.linalg.norm(np.subtract(vec1, vec2))
    else:
        return cossim(vec1, vec2)

def calc_distance_between_words(vectors, word1, word2, distype = 'norm'):
        if word1 in vectors and word2 in vectors:
            if distype is 'norm':
                return np.linalg.norm(np.subtract(vectors[word1], vectors[word2]))
            else:
                return cossim(vectors[word1], vectors[word2])
        return np.nan

def load_models(title, period, align=True):
    models = []
    for year in range(1950,1989, int(period)):
        models.append(KeyedVectors.load_word2vec_format('../embeddings/{}/{}/{}_{}.w2v'.
                                                        format(title, period, year, year+(period-1)),
                                                        binary=True))
    if align == True:
        return align_models(models)
    else:
        return models

def align_models(models):
    models[0].init_sims(replace=True) #l2 normalize
    for i in range(1, len(models)):
        models[i].init_sims(replace=True)
        #models[i] = smart_procrustes_align_gensim(models[i - 1], models[i])
    return models

def embedding_bias(list1, list2, features, models):
    years = [1950, 1960, 1970, 1980]
    means = []
    bounds = []
    values = []
    year = []
    for index, model in enumerate(models):       
        #v1 = np.mean(model[group1_words], axis=0)
        v = calculate_vector(model, male_words, female_words)
        #v2 = np.mean(model[group2_words], axis=0)
        x = []
        #y = []
        for word in features:
            try:
                x.append(cossim(v, model[word]))
                #y.append(cossim(v2, model[word]))
            except:
                pass
        #C = [x_ - y_ for x_, y_ in zip(x, y)]
        values.append(x)
        means.append(np.mean(x))
        bounds.append(pb.bootstrap(x, confidence=0.95, iterations=1000, sample_size=.8, statistic=np.mean))
    return values, means, bounds

def calculate_vectors(models, list1, list2):
    return np.mean([model[word] for word in list1 if word in model.vocab], axis=0) - np.mean([model[word] for word in list2 if word in model.vocab], axis=0)

def calculate_vector(model, list1):
    return np.mean([model[word] for word in list1 if word in model.vocab], axis=0)

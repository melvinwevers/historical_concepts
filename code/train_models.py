import glob
import gensim
import logging
import os

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

path = '../newspapers2018/vk/articles'
path2 = 'sentences'
allFiles = glob.glob(path + "/*")


def getSentencesForYear(year):
    data_path = 'sentences_vk'
    title = 'vk_'
    #sentences = []
    year_path = os.path.join(data_path, str(title) + str(year) + '.txt')
    with open(year_path, 'r') as f:
        sentences = gensim.models.word2vec.LineSentence(f)
    #allFiles = glob.glob(dataFolder + "/*")
    # for file in allFiles:
    #     filename = os.path.basename(file)
    #     if filename.startswith(str(year)):
    #         sentences = gensim.models.word2vec.LineSentence(file)
    # return sentences


def getSentencesInRange(startY, endY):
    return [s for year in range(startY, endY)
            for s in getSentencesForYear(year)]


def train_models():
    model = gensim.models.Word2Vec(min_count=25, size=300,
                                   iter=20, window=10, workers=40, sg=1)

    yearsInModel = 50
    stepYears = 1
    modelFolder = '../models'

    y0 = 1945
    yN = 1995

    for year in range(y0, yN - yearsInModel + 1, stepYears):
        startY = year
        endY = year + yearsInModel
        modelName = modelFolder + '/%d_%d.w2v' % (year, year + yearsInModel)
        print('Building Model: ', modelName)

        corpus = getSentencesInRange(startY, endY)
        bigram_transformer = gensim.models.Phrases(corpus,
                                                   min_count=50, threshold=10)
        bigram = gensim.models.phrases.Phraser(bigram_transformer)
        corpus = list(bigram[corpus])

        model.build_vocab(corpus)
        model.train(corpus, total_examples=model.corpus_count,
                    epochs=model.iter)
        print('....saving')
        model.init_sims(replace=True)
        model.wv.save_word2vec_format(modelName, binary=True)


if __name__ == '__main__':
    train_models()

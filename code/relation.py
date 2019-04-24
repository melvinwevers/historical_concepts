import json
import logging
from gensim.models import Word2Vec, KeyedVectors
import pickle
import numpy as np
import pandas as pd

from itertools import permutations
from sppmimodel import SPPMIModel


def make_df(scores):
        names = []
        accuracies = []
        for i in range(len(scores)):
            print(i)
            name = [v for k,v in scores[i].items() if k =='section']
            correct = sum(len(v)for k, v in scores[i].items() if k == 'correct')
            false = sum(len(v)for k, v in scores[i].items() if k == 'incorrect')
            try:
                accuracy = correct / (correct + false)
            except:
                accuracy = np.nan
            names.append(name)
            accuracies.append(accuracy)
        scores_df = pd.DataFrame(list(zip(names, accuracies)))
        #scores_df.to_pickle('scores_ah_nrc.pkl')

class Relation:
    """A class for making relationship/analogy tests easy"""

    def __init__(self, pathtoset):
        """
        A class which is used to test the accuracy of models viz. some set of questions/predicates.

        :param pathtoset: the path to the predicate set.
        :return: None
        """

        self.pathtoset = pathtoset

    def test_model(self, model):
        """
        Tests a given model with the set.

        :param model: the model for which to test accuracy
        :return: a dictionary with scores per section, and a total score.
        """

        # The most_similar assignment is neccessary because the most_similar function might refer to the original
        # Word2Vec function.
        return model.accuracy(self.pathtoset, most_similar=model.__class__.most_similar, restrict_vocab=None)

    @staticmethod
    def create_set(categories, outfile):
        """
        Creates a test-set .txt file for use in word2vec.
        Conforms to word2vec specs, from the google code repository: https://code.google.com/archive/p/word2vec/

        :param categories: The categories and words in the categories: {NAME: [[tuple_1],[tuple_2],...,[tuple_n]]}
        :param outfile: The file to which to write the text.
        :return: None
        """

        with open(outfile, 'w', encoding='utf8') as f:

            for k, v in categories.items():
                f.write(u": {0}\n".format(k))
                for x in permutations([" ".join(x).lower() for x in v], 2):
                    f.write(u"{0}\n".format(" ".join(x)))

def calculate_analogies(model):
    #sems = json.load(open("data/semtest.json"))
    #Relation.create_set(sems, "data/question-words.txt")
    rel = Relation("data/question-words.txt")
    scores = rel.test_model(model)
    return scores
    #df_scores = make_df(scores)
    #return df_scores
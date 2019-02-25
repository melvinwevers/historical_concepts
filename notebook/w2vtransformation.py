#!/usr/bin/env python
"""Previously called PCAStuff.py"""

import numpy as np


def getModelInv(model):
    """Calculate the inverse matrix of the space defined by the given w2v model"""
    # model.wv.init_sims(replace=False)
    model.init_sims(replace=False)
    # return np.linalg.pinv(model.wv.syn0norm)
    return np.linalg.pinv(model.wv.syn0norm)


def calculateTransform(A, Ainv, B, Binv, sameVocab=False):
    """Given two models (A and B) and their inverse matrices, calculate the
    transformation matrix to go from model A to model B."""
    Nembedded, Nfull = np.shape(Ainv)
    transform = np.zeros([Nembedded, Nembedded])

    for i in range(0, Nembedded):
        # build a unit vector
        v1 = np.zeros(Nembedded)  # vector in space 1
        v1[i] = 1

        # build a vector over the vocabulary A from it, by using the inverse of the W2V mapping
        W1 = V2W(Ainv, v1)

        if sameVocab:
            # user insists the vocabularies are the same, so W2 :== W1
            W2 = W1
        else:
            # as vocabulary A and B do not have to be the same, reconstruct a vector over vocabulary B
            W2 = _wordFromVocab2Vocab(W1, A, B)

        W2 /= np.linalg.norm(W2)

        # find the wordvector in the embbed space again
        v2 = W2V(B, W2)
        v2 /= np.linalg.norm(v2)

        # add it to the transformation matrix
        transform[i, :] = v2
    return transform


def _wordFromVocab2Vocab(W1, A, B):
    """Given two models (A and B), and a word vector from the space defined by
    model A, this function finds the corresponding word vector in space defined
    by B. """
    # vector in space 2
    N = len(B.wv.vocab)
    W2 = np.zeros(N)

    # for each dimension in the vocabulary B
    for j in range(N):
        # find the corresponding word in B
        word = B.wv.index2word[j]

        # if this word is vocabulary A,
        if word in A.wv.vocab:
            # get its index in vocabulary A
            ii = A.wv.vocab[word].index
            # and add its word vector to our result with the proper weight
            W1ii = W1[ii]
            W2[j] = W1ii
        else:
            pass
    return W2


def W2V(M, W):
    """Returns the word vector over the vocabulary a vector in the embedded space"""
    return (M.wv.syn0norm.T).dot(W)


def V2W(Minv, v):
    """Returns the vector in the semantic space to a vector over the vocabulary"""
    return (Minv.T).dot(v)

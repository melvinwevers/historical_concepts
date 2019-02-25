#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2014 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

import logging
import json
import numpy as np
import pyximport; pyximport.install(setup_args={'include_dirs': np.get_include()})
import gensim
import time

from cooccur_matrix import get_cooccur
from gensim import utils, matutils
from gensim.corpora import Dictionary
from sentences import SentenceIter

logger = logging.getLogger(__name__)


class SPPMIFactory(object):
    """
    A class for creating SPPMI matrices out of a raw corpus.
    Based on code by Radim Rehurek: www.github.com/piskvorky/
    """

    @staticmethod
    def _save_freqs(di, outpath):
        """
        Save the word frequencies to a file path as a JSON file.

        :param di: the dictionary.
        :param outpath: the path to which to save the word frequencies.
        :return:
        """

        f = di.dfs
        wordfreqs = {k: f[v] for k, v in di.token2id.items()}

        json.dump(wordfreqs, open(outpath, 'w'))

    @staticmethod
    def _save_sparse_mtr(sparse_mtr, filename):
        """
        Save a sparse matrix to a specified filepath.

        snippet from: http://stackoverflow.com/questions/8955448/save-load-scipy-sparse-csr-matrix-in-portable-data-format

        :param sparse_mtr: the matrix to save.
        :param filename: the filename to which to save the matrix.
        :return:
        """
        np.savez(filename, data=sparse_mtr.data, indices=sparse_mtr.indices,
                 indptr=sparse_mtr.indptr, shape=sparse_mtr.shape)

    @staticmethod
    def _save_word2id(word2id, filename):
        """
        Saves the word2id mapping as a JSON file.

        :param word2id: the word2id mapping.
        :param filename: the filename to which to save.
        :return: None
        """
        json.dump(word2id, open(filename, 'w'))

    @staticmethod
    def create(pathtomapping, pathtocorpus, corpusname, window, numtokeep=50000, save_raw=True, shifts=(1, 5, 10)):
        """
        Creates an Shifted Positive Pointwise Mutual Information matrix.

        :param pathtomapping: The path to the id2word mapping. If this is left empty, the id2word mapping gets
        recreated. Warning: this takes a long time.
        :param pathtocorpus: The path to the corpus folder. The corpus can be spread out over multiple files or folders,
        and is read iteratively.
        :param corpusname: The name of the corpus. Used for saving the files.
        :param window: The window used to consider co-occurrences.
        :param numtokeep: The number of most frequent words to keep. Note that the matrix is non-sparse.
        Because of this, the memory requirements of the code are quadratic.
        :param save_raw: Whether to save the raw co-occurrence matrix as a numpy matrix.
        :param shifts: The shifts to apply to the co-occurrence matrix. Each shifted matrix
        gets saved as a separate model.
        """

        start = time.time()

        if not pathtomapping:
            id2word = Dictionary(SentenceIter(pathtocorpus), prune_at=None)
            id2word.filter_extremes(no_below=5, keep_n=numtokeep)
            id2word.compactify()
            logger.info("Creating the word2id took {0} seconds".format(time.time() - start))
        else:
            id2word = Dictionary.load(pathtomapping)

        inter = time.time()

        word2id = gensim.utils.revdict(id2word)

        corpus = SentenceIter(pathtocorpus)
        raw = get_cooccur(corpus, word2id, window=window)

        logger.info("Creating raw co-occurrence matrix took {0} seconds".format(time.time() - inter))

        if save_raw:
            np.save('{0}-cooccur.npy'.format(corpusname), raw)

        SPPMIFactory._save_word2id(word2id, "{0}mapping.json".format(corpusname))
        SPPMIFactory._save_freqs(id2word, "{0}freqs.json".format(corpusname))

        raw = SPPMIFactory.raw2pmi(raw)

        for k in shifts:
            sparse = SPPMIFactory.shift_clip_pmi(np.copy(raw), k_shift=k)
            SPPMIFactory._save_sparse_mtr(sparse, "{0}-SPPMI-sparse-{1}-shift.npz".format(corpusname, k))
            del sparse

    @staticmethod
    def raw2ppmi(pathtoraw, corpusname, shifts=(1, 5, 10)):
        """
        Creates a PPMI matrix out of a raw co-occurrence matrix.
        First a PMI matrix is created (see raw2pmi, below).
        Any negative entries in this matrix are then truncated to 0 and shifted by a factor of -log(k).

        This function can take multiple shift magnitudes, each of which is performed and saved separately.

        :param pathtoraw: The path to the raw co-occurrence matrix.
        :param corpusname: The name of the corpus.
        :param shifts: A tuple containing shift magnitudes.
        :return: None
        """

        # Create the PMI matrix
        pmi = SPPMIFactory.raw2pmi(np.load(pathtoraw))

        for k in shifts:
            # Shift and clip a copy of the pmi matrix.
            sparse = SPPMIFactory.shift_clip_pmi(np.copy(pmi), k_shift=k)
            # save the PPMI matrix.
            SPPMIFactory._save_sparse_mtr(sparse, "{0}-SPPMI-sparse-{1}-shift.npz".format(corpusname, k))
            del sparse

    @staticmethod
    def raw2pmi(cooccur):
        """
        Computes PMI scores for a matrix of co-occurrence counts.
        All shifts are done in place.

        :param cooccur: The co-occurrence matrix.
        :return: A shifted matrix.
        """

        logger.info("computing PPMI on co-occurence counts")

        # following lines a bit tedious, as we try to avoid making temporary copies of the (large) `cooccur` matrix
        marginal_word = cooccur.sum(axis=1)
        marginal_context = cooccur.sum(axis=0)
        cooccur /= marginal_word[:, None]  # #(w, c) / #w
        cooccur /= marginal_context  # #(w, c) / (#w * #c)
        cooccur *= marginal_word.sum()  # #(w, c) * D / (#w * #c)
        np.log(cooccur, out=cooccur)  # PMI = log(#(w, c) * D / (#w * #c))

        return cooccur

    @staticmethod
    def shift_clip_pmi(pmimtr, k_shift=1.0):
        """
        Turns a pmi matrix into a PPMI matrix by setting all negative values to 0 and then shifting by a factor of
        -log(k).

        :param pmimtr: The matrix of PMI values.
        :param k_shift: The shift factor.
        :return: A PPMI matrix.
        """

        logger.info("shifting PMI scores by log(k) with k=%s" % (k_shift, ))
        pmimtr -= np.log(k_shift)  # shifted PMI = log(#(w, c) * D / (#w * #c)) - log(k)

        logger.info("clipping PMI scores to be non-negative PPMI")
        pmimtr.clip(0.0, out=pmimtr)  # SPPMI = max(0, log(#(w, c) * D / (#w * #c)) - log(k))

        logger.info("normalizing PPMI word vectors to unit length")
        for i, vec in enumerate(pmimtr):
            pmimtr[i] = matutils.unitvec(vec)

        return matutils.corpus2csc(matutils.Dense2Corpus(pmimtr, documents_columns=False)).T

# -*- coding: utf-8 -*-
# The MIT License (MIT)

# Copyright (c) 2014 Julien-Charles Lévesque

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import multiprocessing
import os
from itertools import repeat

import numpy as np

from . import bsgd
from . import util


class BSGDEnsemble():
    '''
    Train a bunch of BSGD SVMs using the budgetedsvm-code base, save
    them to unique model files.
    '''

    def __init__(self, num_classifiers, base_params, num_procs=1,
        output_folder='output'):
        '''
        Parameters:
        -----------
        num_classifiers: number of members of the ensemble (sometimes
         called m).

        base_params: parameters given to the base models.

        num_procs: if greater than one, will parallelize the prediction
         computation with multiprocessing. The predictions of each
         member will be computed in parallel (since they are written
         in files the parallelization is straightforward).

        output_folder: folder in which to store the temporary files
         for predictions. WARNING: The given folder will be emptied
         before starting.
        '''

        self.num_classifiers = num_classifiers
        self.base_params = base_params

        #Parallelization parameters
        self.num_procs = num_procs

        #Setup output folders, give same output folder to base classifiers
        os.makedirs(output_folder, exist_ok=True)
        self.base_params['output_folder'] = output_folder
        self.output_folder = output_folder

        self.base_classifier = bsgd.BudgetSVM
        self.classifiers = [self.base_classifier('_c%i' % i, **base_params)
            for i in range(num_classifiers)]

    def train(self, train_data):
        with multiprocessing.Pool(self.num_procs) as pool:
            pool.map(mp_train,
                zip(range(self.num_classifiers),
                    self.classifiers, repeat(train_data)) )

    def ind_predict(self, X):
        ind_predictions = []
        with multiprocessing.Pool(self.num_procs) as pool:
            ind_predictions = pool.map(mp_get_predictions,
                zip(range(self.num_classifiers),
                    self.classifiers, repeat(X)) )

        ind_predictions = np.array(ind_predictions, dtype=int).T
        return ind_predictions

    def test(self, test_data):
        ind_out = self.ind_predict(test_data)
        labels, _ = util.read_svml_labels(test_data)

        #combine them
        out = majority_vote(ind_out)

        #Boolean stuff.
        acc = accuracy(labels, out)

        print("Testing BSGD ensemble: acc: %.4f. \n" % (acc))

        return acc, out


def mp_train(mp_input):
    i, c, data = mp_input
    print("Training classifier %i." % i)
    c.train(data)
    print("Classifier %i trained." % i)


def mp_get_predictions(mp_input):
    i, c, data = mp_input
    c.test(data)
    p = np.loadtxt(c.prediction_filen)
    return p


def majority_vote(votes):
    u_labels = np.unique(votes)
    n, m = votes.shape
    votes_by_label = np.zeros((n, len(u_labels)))
    for i, v in enumerate(votes):
        votes_by_label[i] = [np.sum(v == u) for u in u_labels]

    out = np.argmax(votes_by_label, axis=1)
    out = np.array([u_labels[o] for o in out])
    return out


def accuracy(target_labels, predicted_labels):
    ''' Computes prediction accuracy.
     '''
    return float(np.sum(target_labels == predicted_labels)) / len(target_labels)

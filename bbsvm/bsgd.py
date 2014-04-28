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

import os
import time
import re
import subprocess

import numpy as np


class BudgetSVM():
    """
    Wrapper around the budgetsvm library as provided by the its
     authors. Requires two executables, budgetsvm-train and
     budgetsvm-predict (will look for them in ..)
    """
    def __init__(self, file_affix='', epochs=1, algorithm=4, L=0.0001,
     budget=100, budget_strategy=0, gamma=-1, bias=False, z=50000,
     output_folder='output', random_seed=-1, max_iters=0):
        '''
        Parameters:
        -----------

        epochs: maximum number of epochs

        algorithm:  0 - pegasos
                    1 - AMM batch
                    2 - AMM online
                    3 - LLSVM
                    4 - BSGD

        L: regularization parameter for SVM updates (lambda)

        gamma: kernel width for RBF (used for BSGD and LLSVM only)

        budget: budget, maximum number of support vectors (BSGD) or
         landmark points (LLSVM)

        budget_strategy: BSGD: 0-removal, 1-merging, LLVSM:0-random,
         1-kmeans, 2-kmedoids

        bias: use a bias term? I think only applied for AMM.
        '''
        self.epochs = epochs
        self.algorithm = algorithm
        self.L = L
        self.budget = budget
        self.budget_strategy = budget_strategy
        self.gamma = gamma
        self.bias = bias
        self.z = z
        #set budgeted-svm to verbose
        self.v = 1

        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)
        self.file_affix = ''

        self.max_iters = int(max_iters) #Will take precedence over the number of epochs

        if random_seed != -1:
            self.random_seed = random_seed
        else:
            self.random_seed = np.random.randint(1000000000)

        self.exec_path = os.path.dirname(__file__)

        self.model_filen = os.path.join(self.output_folder, 'bsgd_model' +
            file_affix)
        self.prediction_filen = os.path.join(self.output_folder, 'bsgd_output' +
            file_affix)

    def train(self, train_data):
        '''
        Requires file names instead of pre-loaded datasets.

        The train function is called by multiprocessing thus it should not
         modify the internal state of the class.
        '''

        d = read_dimensionality(train_data)

        if self.gamma == -1:
            self.gamma = 1 / d

        options = ' -d {} -e {} -A {} -L {} -B {} -M {} -G {} -z {} -v {} -R {} -N {}'.format(d,
            self.epochs, self.algorithm, self.L,
            self.budget, self.budget_strategy, self.gamma,
            self.z, self.v, self.random_seed, self.max_iters)

        #WARNING: Using the same model filename all the time, not thread safe
        #for stuff like replications without changing this.
        cmd = self.exec_path + '/budgetedsvm-train' + options + ' ' + train_data +\
         ' ' + self.model_filen
        cmd = cmd.split(' ')

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE,
         stderr=subprocess.PIPE)

        #If the executions are too close to each other,
        # the random seed will be the same
        time.sleep(1)
        output, error = process.communicate()

        output = output.decode()
        error = error.decode()

        if len(error) > 0:
            raise Exception(error)

        return output

    def test(self, data):
        #always verbose otherwise we don't get testing
        options = '-z {} -v {}'.format(self.z, 1)

        #performance
        cmd = self.exec_path + '/budgetedsvm-predict ' + options + ' ' + data +\
         ' ' + self.model_filen + ' ' + self.prediction_filen
        print(cmd)
        cmd = cmd.split(' ')

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE,
         stderr=subprocess.PIPE)
        output, error = process.communicate()

        output = output.decode()
        error = error.decode()

        if len(error) > 0:
            raise Exception(error)

        #Last line of output contains performance...
        output_lines = output.strip('\n').split('\n')
        perf_str = output_lines[-1]
        perf_str = re.findall(r'\d+.\d+', perf_str)[0]
        accuracy = 1 - float(perf_str) / 100
        print('Tested budgeted SVM on given dataset, accuracy : {}'.format(accuracy))
        return accuracy, output


def read_dimensionality(filename):
    f = open(filename, 'rt')
    #datafile should have the dimensionality written in the first line
    first = f.readline()
    first = first.rstrip()
    if 'DIMENSIONALITY' in first:
        d = int(first.split(' ')[-1])
    else:
        raise Exception('Could not find dimensionality of data. Put it in\
first line of dataset.\n E.g.: #DIMENSIONALITY: 64')
    f.close()
    return d
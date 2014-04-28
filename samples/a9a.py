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

import sys
#Don't need this line if the package was installed.
sys.path.append('../build/lib')

import multiprocessing

import bbsvm.bsgd_ensemble as ens

a9a_ens = ens.BSGDEnsemble(
    num_classifiers=7, #number of classifiers in the ensemble
    base_params={ #base parameters applicable to each BSGD in the ensemble
     'gamma':1e1, #width of the RBF kernel
     'L':1e-3}, #regularization penalty
    num_procs=multiprocessing.cpu_count() #number of processes for training and testing
    )

a9a_ens.train('a9a_train.txt')
a9a_ens.test('a9a_test.txt') #test, will output accuracy score.
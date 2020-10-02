Ensembles of Budgeted Kernel Support Vector Machines
-----------------------

This is a straightforward implementation of the method studied in this paper :


>**Ensembles of Budgeted Kernel Support Vector Machines for Parallel Large Scale Learning.**
> Julien-Charles Lévesque, Christian Gagné and Robert Sabourin.
> Presented in *NIPS 2013 Workshop on Big Learning: Advances in Algorithms and Data Management*, (p. 5),  2013.

Supports the training and computation of predictions in parallel through multiprocessing. More sophisticated approaches could be developed to permit the exploitation of computing clusters.


### Setup / Usage :

Warning: everything was only tested on Linux with python 3.3.

Dependencies:

- make
- setuptools
- numpy
- budgetedsvm toolbox (included)

First build the budgetedsvm-toolbox with the python setup script :

    python setup.py build

Then you can run the sample, `cd` into the samples directory and run the `a9a.py` script, which will train an ensemble of 7 budgeted kernel SVMs on the a9a dataset. If you want to be able to call the `bbsvm` package system-wide, simply install it with `python setup.py install`.

### Paper and abstract :

>In this work, we propose to combine multiple budgeted kernel support vector machines (SVMs) trained with stochastic gradient descent (SGD) in order to exploit large databases and parallel computing resources. The variance induced by budget restrictions of the kernel SVMs is reduced through the averaging of predictions, resulting in greater generalization performance. The variance of the trainings results in a diversity of predictions, which can help explain the better performance. Finally, the proposed method is intrinsically parallel, which means that parallel computing resources can be exploited in a straightforward manner.

[Full paper (PDF)](http://w3.gel.ulaval.ca/~levesq22/papers/bsgdens_biglearn_nips2013.pdf)

### Credits / Acknowledgements

- The budgetedsvm toolbox : [http://www.dabi.temple.edu/budgetedsvm/](http://www.dabi.temple.edu/budgetedsvm/)
- This research was supported through funds from NSERC-Canada. It also benefitted from the computing resources provided by the Calcul Québec/Compute Canada.

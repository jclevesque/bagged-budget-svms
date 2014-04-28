-----------------------------------------------------------
-- BudgetedSVM: A Toolbox for Large-scale Non-linear SVM --
-----------------------------------------------------------

BudgetedSVM is a simple, easy-to-use, and efficient software for large-scale
non-liner SVM classification. This document explains the use of BudgetedSVM.
For MATLAB/OCTAVE interface help please see "./matlab/readme.txt"

Please read the "./GPL.txt" license file before using BudgetedSVM.

Also, the toolbox includes two source files ("./src/libsvmwrite.c" and 
"./src/libsvmread.c") and uses some code from LibSVM package, please 
read "./COPYRIGHT.txt" before using BudgetedSVM for terms and conditions
pertaining to these parts of the toolbox.


Table of Contents
=================
- Table of Contents
- Installation and Data Format
- "budgetedsvm-train" Usage
- "budgetedsvm-predict" Usage
- Examples
- Library Usage
- MATLAB/OCTAVE Interface
- Additional Information


Installation and Data Format
============================
On Unix systems, type `make' to build the `budgetedsvm-train' and `budgetedsvm-predict'
programs. Type 'make clean' to delete the generated files. Run the programs without 
arguments for description on how to use them.

We note that the authors have tested the toolbox on the following platform with success:
> gcc -v
gcc version 4.6.3 (Ubuntu/Linaro 4.6.3-1ubuntu5)

Data format
-----------
The format of training and testing data file is as follows:

<label> <index1>:<value1> <index2>:<value2> ...
.
.
.

Each line contains an instance and is ended by a '\n' character.  For
classification, <label> is an integer indicating the class label
(multi-class is supported). See './a9a_train.txt' for an example. 
For further details about LIBSVM format please see the following webpage
http://www.csie.ntu.edu.tw/~cjlin/libsvm


`budgetedsvm-train' Usage
=======================
In order to get the detailed usage description, run the budgetedsvm-train function
without providing any arguments.

Usage:
budgetedsvm-train [options] train_file [model_file]

Inputs:
options        - parameters of the model
train_file     - url of training file in LIBSVM format
model_file     - file that will hold a learned model
	--------------------------------------------
Options are specified in the following format:
'-OPTION1 VALUE1 -OPTION2 VALUE2 ...'

The following options are available (default values in brackets):

	d - dimensionality of the data (MUST be set by a user)
	A - algorithm, which large-scale SVM solver to use (2):
		0 - Pegasos
		1 - AMM batch
		2 - AMM online
		3 - LLSVM
		4 - BSGD

	e - number of training epochs in AMM and BSGD (5)
	s - number of subepochs (in AMM batch, 1)
	b - bias term in AMM, if 0 no bias added (1.0)
	k - pruning frequency, after how many examples is pruning done in AMM (10000)
	C - pruning aggresiveness, sets the pruning threshold in AMM, OR
			linear-SVM regularization paramater C in LLSVM (10.00000)
	l - limit on the number of weights per class in AMM (20)
	L - learning parameter in AMM and BSGD (0.00010)
	G - kernel width exp(-.5*gamma*||x-y||^2) in BSGD and LLSVM (1/DIMENSIONALITY)
	B - total SV set budget in BSGD, OR number of landmark points in LLSVM (100)
	M - budget maintenance strategy in BSGD (0 - removal; 1 - merging), OR
			landmark selection in LLSVM (0 - random; 1 - k-means; 2 - k-medoids) (1)

	z - training and test file are loaded in chunks so that the algorithms can
			handle budget files on weaker computers; z specifies number of examples
			loaded in a single chunk of data (50000)
	w - model weights are split in chunks, so that the algorithm can handle
			highly dimensional data on weaker computers; w specifies number of
			dimensions stored in one chunk (1000)
	S - if set to 1 data is assumed sparse, if 0 data assumed non-sparse, used to
			speed up kernel computations (default is 1 when percentage of non-zero
			features is less than 5%, and 0 when percentage is larger than 5%)
	v - verbose output; 1 to show the algorithm steps, 0 for quiet mode (0)
	--------------------------------------------

The model is saved in a text file which has the following rows:
[ALGORITHM, DIMENSION, NUMBER OF CLASSES, LABELS, NUMBER OF WEIGHTS, BIAS TERM, KERNEL WIDTH, MODEL] 
In order to compress memory and to use the memory efficiently, we coded the model in the following way:

For AMM batch, AMM online, PEGASOS:	The model is stored so that each row of the text file corresponds 
to one weight. The first element of each weight is the class of the weight, followed by the degradation 
of the weight. The rest of the row corresponds to non-zero elements of the weight, given as 
feature_index:feature_value, in a standard LIBSVM format.

For BSGD: The model is stored so that each row corresponds to one support vector (or weight). The 
first elements of each weight correspond to alpha parameters for each class, given in order of 
"labels" member of the Matlab structure. However, since alpha can be equal to 0, we use LIBSVM format
to store alphas, as -class_index:class-specific_alpha, where we added '-' (minus sign) in front of 
the class index to differentiate between class indices and feature indices that follow. After the 
alphas, in the same row the elements of the weights (or support vectors) for each feature are given 
in LIBSVM format.

For LLSVM: The model is stored so that each row corresponds to one landmark point. The first element of 
each row corresponds to element of linear SVM hyperplane for that particular landmark point. This is 
followed by features of the landmark point in the original feature space of the data set in LIBSVM format. 


`budgetedsvm-predict' Usage
=========================
In order to get the detailed usage description, run the budgetedsvm-predict function
without providing any arguments.

Usage:
budgetedsvm-predict [options] test_file model_file output_file

Inputs:
options        - parameters of the model
test_file      - url of test file in LIBSVM format
model_file     - file that holds a learned model
output_file    - url of file where output will be written
--------------------------------------------
Options are specified in the following format:
'-OPTION1 VALUE1 -OPTION2 VALUE2 ...'

The following options are available (default values in brackets):

	z - the training and test file are loaded in chunks so that the algorithm can
			handle budget files on weaker computers; z specifies number of examples
			loaded in a single chunk of data (50000)
	w - the model weight is split in parts, so that the algorithm can handle
			highly dimensional data on weaker computers; w specifies number of
			dimensions stored in one chunk (1000)
	S - if set to 1 data is assumed sparse, if 0 data assumed non-sparse, used to
			speed up kernel computations (default is 1 when percentage of non-zero
			features is less than 5%, and 0 when percentage is larger than 5%)
	v - verbose output; 1 to show algorithm steps, 0 for quiet mode (0)
--------------------------------------------


Examples
========
Here is a simple example on how to train and test a classifier on the provided adult9a data set,
after budgetedsvm-train and budgetedsvm-predict functions were compiled by running 'make'.

> bin/budgetedsvm-train -A 1 -L 0.001 -e 5 -d 123 -v 1 a9a_train.txt
> bin/budgetedsvm-predict -v 1 a9a_test.txt a9a_train.txt.model a9a_preds.txt

Note that the train and predict programs were created in the "./bin" folder, which is
why we appended "bin/" to the calls to the functions. If you run the programs in Windows,
a user should use "\" (back-slash) instead of "/" (forward-slash) when specifying the path
to the programs in the command prompt.

The first command uses AMM batch ("-A 1") algorithm to train multi-hyperplane
machine for 5 epochs ("-e 5"), using lambda = 0.001 ("-L 0.001"). As adult9a data 
set is of dimensionality 123, we also write "-d 123", and choose verbose output
("-v 1") which prints detailed steps of the algorithm. As we did not specify a name
for the model file, it will be created such that suffix '.model' is appended to the
filename of the training file.
The second command tests the model on testing data set, and prints the 
accuracy on the testing set while saving the predictions to a9a_preds.txt. We
also set verbose output by writing "-v 1".


Library Usage
=============
See the "doc/BudgetedSVM reference manual.pdf" or open "doc/html/index.html" in your browser 
for details about the implementation.


MATLAB/OCTAVE Interface
=======================
Please check the README.txt file in the "./matlab" directory for more information.


Additional Information
======================
The toolbox was written by Nemanja Djuric, Liang Lan, and Slobodan Vucetic
from the Department of Computer and Information Sciences, Temple University,
together with Zhuang Wang from Siemens Corporate Research & Technology.

For any questions, please contact Nemanja Djuric at <nemanja.djuric@temple.edu>.

Acknowledgments:
This work was supported in part by the National Science 
Foundation via the grant NSF-IIS-0546155.
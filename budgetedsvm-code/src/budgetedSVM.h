/*!
	\file budgetedSVM.h
	\brief Header file defining classes and functions used throughout the budgetedSVM toolbox.
*/
/*
	* This program is a free software; you can redistribute it and/or modify
	* it under the terms of the GNU General Public License as published by
	* the Free Software Foundation; either version 3 of the License, or
	* (at your option) any later version.

	Authors	:	Nemanja Djuric
	Name	:	budgetedSVM.h
	Date	:	November 29th, 2012
	Desc.	:	Header file defining classes and functions used throughout the BudgetedSVM toolbox.
	Version	:	v1.01
*/

#ifndef _BUDGETEDSVM_H
#define _BUDGETEDSVM_H

#ifdef __cplusplus
extern "C" {
#endif

/*!
    \brief Large (infinite) value, similar to Matlab's Inf.
*/
#define INF HUGE_VAL

/*!
    \brief Defines pointer to a function that prints information for a user, defined for more clear code.
*/
typedef void (*funcPtr)(const char * text);

/*! \fn void svmPrintString(const char* text)
	\brief Prints string to the output.
	\param [in] text Text to be printed.

	Prints string to the output. Exactly to which output should be specified by \link setPrintStringFunction\endlink, which modifies the callback that is invoked for printing. This is convinient when simple printf() can not be used, for example if we want to print to Matlab prompt. For example on how to set the printing function in Matlab environment, see the implementation of \link parseInputMatlab\endlink.
*/
void svmPrintString(const char* text);

/*! \fn void setPrintStringFunction(funcPtr printFunc)
	\brief Modifies a callback that prints a string.
	\param [in] printFunc New text-printing function.

	This function is used to modify the function that is used to print to standard output.
	After calling this function, which modifies the callback function for printing, the text is printed simply by invoking \link svmPrintString\endlink. \sa funcPtr
*/
void setPrintStringFunction(funcPtr printFunc);

/*! \fn void svmPrintErrorString(const char* text)
	\brief Prints error string to the output.
	\param [in] text Text to be printed.

	Prints error string to the output. Exactly to which output should be specified by \link setPrintErrorStringFunction\endlink, which modifies the callback that is invoked for printing. This is convinient when an error is detected and, prior to printing appropriate message to a user, we want to exit the program. For example on how to set the printing function in Matlab environment, see the implementation of \link parseInputMatlab\endlink.
*/
void svmPrintErrorString(const char* text);

/*! \fn void setPrintErrorStringFunction(funcPtr printFunc)
	\brief Modifies a callback that prints an error string.
	\param [in] printFunc New text-printing function.

	This function is used to modify the function that is used to print to error output.
	After calling this function, which modifies the callback function for printing error string, the text is printed simply by invoking \link svmPrintErrorString\endlink. \sa funcPtr
*/
void setPrintErrorStringFunction(funcPtr printFunc);

/*! \fn bool fgetWord(FILE *fHandle, char *str)
	\brief Reads one word string from an input file.
	\param [in] fHandle Handle to an open file from which one word is read.
	\param [out] str A character string that will hold the read word.
	\return True if end-of-line or end-of-file encountered after reading a word string, otherwise false.

	The function is similar to C++ functions fgetc() and getline(), only that it reads a single word from a text file. For the purposes of this project, a word is defined as a sequence of characters that does not contain a white-space character or new-line character '\n'. As a model in BudgetedSVM is stored in a text file where each line may corresponds to a single support vector, it is also useful to know if we reached the end of the line or the end of the file, which is indicated by the return value of the function.
*/
bool fgetWord(FILE *fHandle, char *str);

/*! \fn bool readableFileExists(const char fileName[])
	\brief Checks if the file, identified by the input parameter, exists and is available for reading.
	\param [in] fileName Handle to an open file from which one word is read.
	\return True if the file exists and is available for reading, otherwise false.
*/
bool readableFileExists(const char fileName[]);


/*!
    \brief Available large-scale, non-linear algorithms (note: unlike other algorithms, PEGASOS is a linear SVM solver).
*/
enum {PEGASOS, AMM_BATCH, AMM_ONLINE, LLSVM, BSGD};


/*! \struct parameters
    \brief Structure holds the parameters of the implemented algorithms.

	Structure holds the parameters of the implemented algorithms. If needed, the default parameters for each algorithm can be manually modified here.
*/
struct parameters
{
	/*! \var unsigned int ALGORITHM
		\brief Algorithm that is used, 0 - Pegasos; 1 - AMM batch; 2 - AMM online; 3 - LLSVM; 4 - BSGD (default: 2)
	*/
	/*! \var unsigned int DIMENSION
		\brief Dimensionality of the classification problem, MUST be set by a user (default: 0)

		Although the dimensionality of the data set can be found from the training data set during loading, we ask a user to specify it beforehand, as it is usually a known parameter.
		The reason why we require this as an input is to speed up processing of the data, since the emphasis of the software is on speeding up the training of classification algorithm on
		large data, and this little piece of information can help avoid unnecessary bookkeeping tasks. More specifically, the parameter is important for memory management of
		\link budgetedVector\endlink, where it is used to find how many weight chunks of size \link CHUNK_WEIGHT\endlink are needed to represent the data.

		However, in the case of Matlab interface, it is not required to manually set this parameter as it is easily found by reading the dimensions of the
		Matlab structure holding the data set.
	*/
	/*! \var unsigned int NUM_EPOCHS
		\brief Number of training epochs (default: 5)

		Number of times the data set is seen by the training procedure, each time randomly reshuffled.
	*/
	/*! \var unsigned int NUM_SUBEPOCHS
		\brief Number of training subepochs of AMM batch algorithm (default: 1)

		AMM batch has an option to reassign data points to weights several times during one epoch. In the most extreme case, if \link NUM_SUBEPOCHS\endlink is equal to the
		size of the data set, we obtain AMM online algorithm. This parameter specifies how many times we reassign points to weights within a single epoch.
	*/
	/*! \var double BIAS_TERM
		\brief Bias term of AMM batch, AMM online, and PEGASOS algorithms (default: 1.0)

		If the parameter is non-zero, a bias, or intercept term, is added to the data set as an additional feature. The value of this additional feature is equal to BIAS_TERM.
	*/
	/*! \var unsigned int K_PARAM
		\brief Frequency of weight pruning of AMM algorithms (default: 10,000 iterations)

		In order to reduce the complexity of the learned model, which directly improves generalization of the model as shown in the AMM paper, pruning of small non-zero weights is performed.
		\link K_PARAM \endlink specifies the frequency of weight pruning, i.e., after how many iterations we perform the pruning step. Aggressiveness of pruning is controlled
		by \link C_PARAM \endlink parameter.
	*/
	/*! \var double C_PARAM
		\brief Weight pruning parameter of AMM algorithms, OR linear-SVM regularization parameter used in LLSVM (default: 10.0)

		- AMM: In order to reduce the complexity of the learned model, which directly improves generalization of the model as shown in the AMM paper, pruning of small non-zero weights is performed.
		C_PARAM specifies the aggressiveness of weight pruning, where larger value results in pruning of more weights. More specifically, we sort the weights by their L2-norms,
		and then prune from the smallest toward larger weight until the cummulative weight norm exceeds value of C_PARAM. Frequency of pruning is controlled by \link K_PARAM \endlink parameter.

		- LLSVM: Regularization parameter used in the objective function of SVM, which is used in the LLSVM algorithm to find the best hyperplane separating the classes after mapping from the input feature space to the new, linearized feature space. Larger values result in more complex model.
	*/
	/*! \var unsigned int CHUNK_WEIGHT
		\brief Size of chunk of \link budgetedVector \endlink weight (whole vector is split into smaller parts) (default: 1,000)

		While \link CHUNK_SIZE \endlink helps when one is working with large data with many data points, this parameter helps when working with high-dimensional data.
		When the data is sparse, then we do not have to explicitly store every feature as most of them are equal to 0. One option is simply to follow LIBSVM format, and
		store a vector in two linked lists, one holding feature index and the other holding the corresponding feature value. However, we found that accessing this data structure can become prohibitively slow, as for
		high-dimensional data weights can become less sparse than the original data due to the weight update process. For example, when we want to update a specific feature during gradient descent
		training we would like to do it very quickly, most preferably we would like to have random access to the element of the weight vector that will be updated. We address this by storing a
		vector into linked list, where each element of the linked list, called <em>weight chunk</em>, holding a subset of features. For example, the first chunk would hold features indexed from 1 to
		CHUNK_SIZE, the second would hold features indexed from CHUNK_SIZE+1 to 2*CHUNK_SIZE, and so on. If all elements of a weight chunk are zero, we do not allocate memory for that array.
		In our experience, this significantly improved the training and testing time on truly high-dimensional data, such as on URL data set with more than 3.2	million features. If \link CHUNK_WEIGHT \endlink is equal to 1, we obtain the LIBSVM-type representation.
	*/
	/*! \var unsigned int CHUNK_SIZE
		\brief Size of the chunk of the data loaded at once (default: 50,000 data points)

		While \link CHUNK_WEIGHT \endlink helps when one is working with high-dimensional data, this parameter helps when working with large data with many instances.
		If the data set is very large and can not fit into memory, we can then load only a small part of it (called <em>data chunk</em>), that is processed before being discarded to make room for the next
		chunk. Therefore, we load only a smaller part of the large data set, with size of this chunk specified by this parameter.
	*/
	/*! \var unsigned int LIMIT_NUM_WEIGHTS_PER_CLASS
		\brief Maximum number of weight per class of AMM algorithms (default: 20)

		As the number of weights in AMM algorithms is infinite, we can set the limit on the number of non-zero weights that can be stored in memory. This can be done in order
		to avoid memory-related problems. Once the limit is reached, we do not allow creation of new non-zero weights until some get pruned.
	*/
	/*! \var unsigned int BUDGET_SIZE
		\brief Size of the budget of BSGD algorithm, OR number of landmark points in LLSVM algorithm (default: 100)

		- BSGD: Maximum number of support vectors that can be stored. After the budget is exceeded, \link MAINTENANCE_SAMPLING_STRATEGY\endlink specifies how the number of support vectors is kept limited.

		- LLSVM: In addition, it also specifies the number of landmark points in LLSVM algorithm, that are used to represent the data set in lower-dimensional space using the Nystrom method.
	*/
	/*! \var unsigned int MAINTENANCE_SAMPLING_STRATEGY
		\brief Budget maintenance strategy of BSGD algorithm, 0 - random removal; 1 - merging, OR type of landmark points sampling in LLSVM algorithm, 0 - random; 1 - k-means; 2 - k-medoids (default: 0)

		- BSGD: Whenever a number of support vectors in BSGD algorithm exceeds \link BUDGET_SIZE\endlink, one of the following budget maintenance steps is performed, depending on the value of the MAINTENANCE_SAMPLING_STRATEGY parameter
			- 0 - deleting random support vector to maintain the budget
			- 1 - take two support vectors and merging them into one. The new, merged support vector is located on the straight line connecting the two existing support vectors; where exactly on the line is explained in \em computeKmax() function from \em bsdg.cpp file. Then, the two existing support vectors are deleted and the merged vector is inserted in the budget. (default setting)

		- LLSVM: Specifies how the landmark points, used to represent the data set in lower-dimensional space using the Nystrom method, are chosen.
			- 0 - landmark points are randomly sampled from the the first loaded data chunk
			- 1 - landmark points will be cluster centers after running k-means on the first loaded data chunk (default setting)
			- 2 - landmark points will be cluster medoids after running k-medoids on the first loaded data chunk
	*/
	/*! \var unsigned int K_MEANS_ITERS
		\brief Number of k-means iterations in initialization of LLSVM algorithm (default: 10)

		In order to find better lower-dimensional representation of the data set using Nystrom method, k-means can be used to improve the choice of landmark points. Unlike in random sampling
		of landmark points from the data set, cluster centers of k-means will represent \link BUDGET_SIZE\endlink points used for the Nystrom method.
	*/
	/*! \var double GAMMA_PARAM
		\brief Kernel width parameter in Gaussian kernel exp(-0.5 * GAMMA_PARAM * ||x - y||^2) (default: 1/DIMENSIONALITY)
	*/
	/*! \var double LAMBDA_PARAM
		\brief Lamdba parameter of AMM algorithms (default: 0.0001)

		The parameter defines the level of model regularization of AMM models, where larger values result in less complex model. The parameter is similar to \link C_PARAM\endlink parameter used in LLSVM when solving linear-SVM, as both parameters are used to set the level of regularization, only with opposite effect. Namely, while larger values of LAMBDA_PARAM result in less complex AMM model, larger values of \link C_PARAM\endlink result in more complex LLSVM model.
	*/
	/*! \var bool VERBOSE
		\brief Print verbose output during algorithm execution, 1 - verbose output; 0 - quiet (default: 0)
	*/
	/*! \var unsigned int VERY_SPARSE_DATA
		\brief User set parameter, if a user believes the data is very sparse this parameters can be set to 0/1, where 1 - very sparse data; 0 - not very sparse data (default: see long description)

		When computing the kernels between support vectors/hyperplanes kept in the available budget in budgetedVector objects on one side, and the incoming data points on the other, we have two options: (1) we can either do the computations directly between the support vectors and data points that are stored in budgetedData; or (2) we can do the computations between the support vectors and data points that are in the intermediate step stored in the budgetedVector object. When the data is very sparse option (1) is faster, as there is very small number of non-zero features that affects the speed of the computations, and the overhead of creating the budgetedVector instance might prove too costly. On the other hand, when the data is not too sparse, then it might prove faster to first create budgetedVector that will hold the incoming data point, and only then do the kernel computations. The reason is partly in a slow modulus operation that is used in the case (1) (please refer to the implementation of linear and Gaussian kernels to see how it was coded).

		If a user does not manually set this parameter to 0 (i.e., instructs the toolbox to compute kernels as in case (1)) or 1 (i.e., compute kernels as in case (2)), the default setting will be 0 if the sparsity of the loaded data is less than 5% (i.e., less than 5% of the features are non-zero on average), otherwise it will default to 1. For this default behavior that is adaptive to the found data sparsity a developer can set this parameter to anything other than 0 or 1. For more details, please see the train and test functions of the implemented algorithms, and look for code parts where VERY_SPARSE_DATA appears. \sa updateVerySparseDataParameter(), budgetedVector::linearKernel(unsigned int, budgetedData*, parameters*), budgetedVector::linearKernel(budgetedVector*), budgetedVector::gaussianKernel(unsigned int, budgetedData*, parameters*, long double), budgetedVector::gaussianKernel(budgetedVector*, parameters*)
	*/

	unsigned int ALGORITHM,	NUM_SUBEPOCHS, NUM_EPOCHS, K_PARAM, DIMENSION, CHUNK_SIZE, CHUNK_WEIGHT,
		LIMIT_NUM_WEIGHTS_PER_CLASS, BUDGET_SIZE, K_MEANS_ITERS, MAINTENANCE_SAMPLING_STRATEGY, VERY_SPARSE_DATA, NUM_ITERS;
	double       C_PARAM, BIAS_TERM, GAMMA_PARAM, LAMBDA_PARAM;
	bool         VERBOSE;

	/*! \fn parameters(void)
		\brief Constructor of the structure. The default values of the parameters can be modified here manually.
	*/
	parameters(void)
	{
		////// configurable parameters - modified by input option string (default values set here)
		// algorithm parameters
		ALGORITHM						= AMM_ONLINE;	// algorithm to use (check enumeration defined before this structure)
		NUM_EPOCHS                  	= 5;        	// number of scans through training data
		BIAS_TERM                   	= 1.0;      	// bias term, if 0 no bias added
		DIMENSION                   	= 0;        	// dataset dimensionality (must be set by user)

		K_PARAM                     	= 10000;    	// pruning frequency, pruning done every K_PARAM points
		C_PARAM                     	= 10.0;     	// pruning aggresiveness (i.e., the higher value, the more weights are pruned), OR regularization parameter in LLSVM
		NUM_SUBEPOCHS               	= 1;        	// number of subepochs of AMM_batch
		LAMBDA_PARAM                	= 0.0001;   	// learning (lambda) parameter in AMM

		BUDGET_SIZE                		= 100;      	// SVM budget size in BSGD, OR number of landmark points in LLSVM
		MAINTENANCE_SAMPLING_STRATEGY	= 1;        	// 0 - smallest removal or 1 - merging maintenance in BSGD, OR sampling of landmark points in LLSVM: 0 - random; 1 - k-means; 2 - k-medoids
		GAMMA_PARAM                 	= 0.0;      	// sigma scale parameter in Gaussian kernel
		K_MEANS_ITERS					= 10;			// number of k-means / k-medoids iterations during initialization

		VERBOSE							= 0;        	// verbose output
		CHUNK_SIZE						= 50000;    	// size of chunk of file loaded in budgetedData (when a file is too budget to fit in memory)
		CHUNK_WEIGHT                	= 1000;     	// size of chunk of budgetedVector weight (since vector is split into many small parts)
		LIMIT_NUM_WEIGHTS_PER_CLASS 	= 20;       	// maximum number of weights per class
		VERY_SPARSE_DATA				= 99;        	// for very sparse data, we can speed up computations by directly computing kernels from budgetedData, 99 for default, 0 and 1 when set by user;
														// 		0 when a user wants all kernel computations done between budgetedVectors, 1 for computations between budgetedVectors and vectors
														//		stored in the input budgetedData; when the data is very sparse directly computing kernels with data points stored in budgetedData on
														//		one side and vectors stored in budgetedVector may be faster, as compared to computing kernels between two budgetedVectors; the default
														// 		is VERY_SPARSE_DATA = 1 if data sparsity is less than 5%, otherwise 0
        NUM_ITERS = 0;
		////// end of configurable parameters
	};

	/*! \fn void updateVerySparseDataParameter(double dataSparsity)
		\brief If \link VERY_SPARSE_DATA \endlink parameter was not set by a user, this function sets this parameter according to the sparsity of the loaded data.
		\param [in] dataSparsity The sparsity of the loaded data set.

		When computing the kernels between support vectors/hyperplanes kept in the available budget in budgetedVector objects on one side, and the incoming data points on the other, we have two options: (1) we can either do the computations directly between the support vectors and data points that are stored in budgetedData; or (2) we can do the computations between the support vectors and data points that are in the intermediate step stored in the budgetedVector object. When the data is very sparse option (1) is faster, as there is very small number of non-zero features that affects the speed of the computations, and the overhead of creating the budgetedVector instance might prove too costly. On the other hand, when the data is not too sparse, then it might prove faster to first create budgetedVector that will hold the incoming data point, and only then do the kernel computations. The reason is partly in a slow modulus operation that is used in the case (1) (please refer to the implementation of linear and Gaussian kernels to see how it was coded. \sa VERY_SPARSE_DATA, budgetedVector::linearKernel(unsigned int, budgetedData*, parameters*), budgetedVector::linearKernel(budgetedVector*), budgetedVector::gaussianKernel(unsigned int, budgetedData*, parameters*, long double), budgetedVector::gaussianKernel(budgetedVector*, parameters*)
	*/
	void updateVerySparseDataParameter(double dataSparsity)
	{
		// if the parameter is already set then just return and change nothing; it can be that it was set by a user
		// 	or it was already set when the earlier data chunks were loaded
		if ((VERY_SPARSE_DATA == 0) || (VERY_SPARSE_DATA == 1))
			return;

		// if the sparsity is less than 5%, then we say that we are working with very sparse data
		if (dataSparsity < 5.0)
			VERY_SPARSE_DATA = 1;
		else
			VERY_SPARSE_DATA = 0;
	};
};

/*! \class budgetedData
    \brief Class which handles manipulation of large data sets that cannot be fully loaded to memory (using a data structure similar to Matlab's sparse matrix structure).

	In order to handle large data sets, we do not load the entire data into memory, instead load it in smaller chunks. The loaded chunk is stored in a structure similar to Matlab's sparse matrix structure. Namely, only non-zero features and corresponding feature values of data points are stored in one budget vector for fast access, with additional vector that hold pointers to feature vector telling us where the information for each individual data point starts.
*/
class budgetedData
{
	/*! \var FILE *ifile
		\brief Pointer to a FILE object that identifies input data stream.
	*/
	/*! \var FILE *fAssignFile
		\brief Pointer to a FILE object that identifies data stream of current assignments, used for AMM batch algorithm.

		During AMM batch training phase we need to keep track of which non-zero weight is assigned to which data point. We store the assignments into text file and load them together with
		the data chunk currently loaded, as it might be to expensive to store all assignments in memory. In order to keep track of this weight-example mapping, each weight vector
		also has a unique \link budgetedVector::weightID\endlink, assigned to each vector upon creation. \sa parameters::CHUNK_SIZE \sa budgetedVector::weightID
	*/
	/*! \var const char* ifileName
		\brief Filename of LIBSVM-style .txt file with input data.
	*/
	/*! \var const char* ifileNameAssign
		\brief Filename of .txt file that keeps current assignments of weights to input data points, used for AMM batch algorithm.

		During AMM batch training phase we need to keep track of which non-zero weight is assigned to which data point. We store the assignments into text file and load them together with
		the data chunk currently loaded, as it might be to expensive to store all assignments in memory. \sa parameters::CHUNK_SIZE
	*/
	/*! \var long dimension
		\brief Dimensionality of the input data, does not include one additional dimension due to possible non-zero bias term. \sa parameters::BIAS_TERM
	*/
	/*! \var long dimensionHighestSeen
		\brief Highest dimension seen during loading of the data, should always be smaller than the set dimension. Used to detect badly specified data dimension.
	*/
	/*! \var bool fileOpened
		\brief Indicates that the input data .txt file is open.
	*/
	/*! \var bool fileAssignOpened
		\brief Indicates that the .txt file with current assignments is open, used for AMM batch algorithm.
	*/
	/*! \var bool dataPartiallyLoaded
		\brief Indicates that the data is only partially loaded to memory. It can also be fully loaded, e.g., when using data already loaded by some other application, Matlab for instance.
	*/
	/*! \var bool keepAssignments
		\brief Indicates that assignments should be kept, true only for AMM batch algorithm.
	*/
	/*! \var unsigned long loadTime
		\brief Measures the time spent to load the data.
	*/

	/*! \var char* al
		\brief Array of labels of the current data chunk, always of length \link N\endlink.
	*/
	/*! \var vector <long> aj
		\brief Vector of indices of non-zero features of data points of the current data chunk. Where the data points start and end in this vector is specified by \link ai \endlink vector.
	*/
	/*! \var vector <float> an
		\brief Vector of non-zero features of data points of the current data chunk. Where the data points start and end in this vector is specified by \link ai \endlink vector.
	*/
	/*! \var vector <long> ai
		\brief Vector that tells us where the data point starts in vectors \link an \endlink and \link aj\endlink, always of length \link N\endlink.
	*/
	/*! \var vector <int> yLabels
		\brief Vector of possible labels, either found during loading or initialized during testing phase by the learned model.
	*/
	/*! \var unsigned int N
		\brief Number of data points loaded.
	*/
	/*! \var unsigned int *assignments
		\brief Assignments for the current data chunk, used for AMM batch algorithm. \sa fAssignFile
	*/
	/*! \var unsigned int numNonZeroFeatures
		\brief Number of non-zero features of the currently loaded chunk, found during loading of the data. Used to compute the sparsity of the data.
	*/
	/*! \var unsigned int loadedDataPointsSoFar
		\brief Total number of data points loaded so far.
	*/

	protected:
		FILE *ifile, *fAssignFile;
		const char* ifileName, *ifileNameAssign;
		unsigned int dimension;
		unsigned int dimensionHighestSeen;
		unsigned int numNonZeroFeatures, loadedDataPointsSoFar;
		bool fileOpened, fileAssignOpened, dataPartiallyLoaded, keepAssignments;

	public:
		unsigned long loadTime;				// keeps track of time spent loading data
		vector <float> an;       			// feature value
		vector <unsigned int> aj, ai;    	// where the example starts, feature number
		unsigned char* al;                	// example labels
		vector <int> yLabels;          		// list of possible labels
		unsigned int N;            			// number of examples loaded in memory
		unsigned int *assignments;			// assignments to examples (used in AMM batch only)


		/*! \fn double getSparsity(void)
			\brief Get the sparsity of the data set (i.e., percentage of non-zero features). It is a number between 0 and 100, showing the sparsity in percentage points.
			\return Returns the sparsity of the data set in percentage points.
		*/
		double getSparsity(void)
		{
			return (100.0 * (double) numNonZeroFeatures / ((double) loadedDataPointsSoFar * (double) dimension));
		};

		/*! \fn unsigned int getNumLoadedDataPointsSoFar(void)
			\brief Get total number of data points loaded since the beginning of the epoch.
			\return Number of data points loaded since the beginning of the epoch.
		*/
		unsigned int getNumLoadedDataPointsSoFar(void)
		{
			return loadedDataPointsSoFar;
		};

		/*! \fn budgetedData(bool keepAssignments = false, vector <int> *yLabels = NULL)
			\brief Vanilla constructor, just initializes the variables.
			\param [in] keepAssignments True for AMM batch, otherwise false. File 'temp_assigns.txt' will be created and deleted to keep the assignments.
			\param [in] yLabels Possible labels in the classification problem, for training data is NULL since they are inferred from data.
		*/
		budgetedData(bool keepAssignments = false, vector <int> *yLabels = NULL);


		/*! \fn budgetedData(const char fileName[], unsigned int dimension, unsigned int chunkSize, bool keepAssignments = false, vector <int> *yLabels = NULL)
			\brief Constructor that takes the data from LIBSVM-style .txt file.
			\param [in] fileName Path to the input .txt file.
			\param [in] dimension Dimensionality of the classification problem.
			\param [in] chunkSize Size of the input data chunk that is loaded.
			\param [in] keepAssignments True for AMM batch, otherwise false. File 'temp_assigns.txt' will be created and deleted to keep the assignments.
			\param [in] yLabels Possible labels in the classification problem, for training data is NULL since inferred from data.
		*/
		budgetedData(const char fileName[], unsigned int dimension, unsigned int chunkSize, bool keepAssignments = false, vector <int> *yLabels = NULL);

		/*! \fn virtual ~budgetedData(void)
			\brief Destructor, cleans up the memory.
		*/
		virtual ~budgetedData(void);

		/*! \fn void saveAssignment(unsigned int *assigns)
			\brief Saves the current assignments, used by AMM batch.
			\param [in] assigns Current assignments.
		*/
		void saveAssignment(unsigned int *assigns);

		/*! \fn void readChunkAssignments(bool endOfFile)
			\brief Reads assignments for the current chunk, used by AMM batch.
			\param [in] endOfFile If the final chunk, close the assignment file.

			During AMM batch training phase we need to keep track of the assignment of non-zero weights to data points. We store the assignments into a text file and load them together with the data chunk currently loaded, as it may be to expensive to store all assignments in memory when working with large data sets.
		*/
		void readChunkAssignments(bool endOfFile);

		/*! \fn void flushData(void)
			\brief Clears all data taken up by the current chunk.
		*/
		void flushData(void);


		/*! \fn virtual bool readChunk(unsigned int size, bool assign = false)
			\brief Reads the next data chunk.
			\param [in] size Size of the chunk (i.e., number of data points) to be loaded.
			\param [in] assign True if assignments should be saved, false otherwise.
			\return True if just read the last data chunk, false otherwise.

			In order to handle large data sets, we do not load the entire data into memory, instead load it in smaller chunks. Once we have finished processing a loaded data chunk, we load a new one using this function. The return value tells us if there are more chunks left; while there is still data to be loaded the function returns false, if we are done with the data set the function returns true. In the case of the AMM_batch algorithm, we also need to store current assignments of data points to weights, if the input "assign" is true then the function also initializes a .txt file for purpose of storing these assignments when the first chunk is loaded.
		*/
		virtual bool readChunk(unsigned int size, bool assign = false);

		/*! \fn float getElementOfVector(unsigned int vector, unsigned int element)
			\brief Returns an element of a vector stored in \link budgetedData\endlink structure.
			\param [in] vector Index of the vector (C-style indexing used, starting from 0; note that LibSVM format indices start from 1).
			\param [in] element Index of the element of the vector (C-style indexing used, starting from 0; note that LibSVM format indices start from 1).
			\return Element of the vector specified as an input.

			In the case that we need to read an element of a vector from currently loaded data chunk, we can use this function to access these vector elements.
		*/
		float getElementOfVector(unsigned int vector, unsigned int element);

		/*! \fn long double getVectorSqrL2Norm(unsigned int vector, parameters *param)
			\brief Returns a squared L2-norm of a vector stored in \link budgetedData\endlink structure.
			\param [in] vector Index of the vector (C-style indexing used, starting from 0; note that LibSVM format indices start from 1).
			\param [in] param The parameters of the algorithm.
			\return Squared L2-norm of a vector.

			This function returns squared L2-norm of a vector stored in the \link budgetedData\endlink structure. In particular, it is used to speed up the computation of Gaussian kernel.
		*/
		long double getVectorSqrL2Norm(unsigned int vector, parameters *param);

		/*! \fn double distanceBetweenTwoPoints(unsigned int index1, unsigned int index2)
			\brief Computes Euclidean distance between two data points from the input data.
			\param [in] index1 Index of the first data point.
			\param [in] index2 Index of the second data point.
			\return Euclidean distance between the two points.
		*/
		double distanceBetweenTwoPoints(unsigned int index1, unsigned int index2);
};

/*! \class budgetedVector
    \brief Class which handles high-dimensional vectors.

	In order to handle high-dimensional vectors (i.e., data points), we split the data vector into an array of smaller vectors (or chunks; implemented as a vector of arrays), and allocate memory
	for each chunk only if it contains at least one element that is non-zero. This is especially beneficial for very sparse data sets, where we can have considerable
	memory gains. Each chunk has a pointer to it stored in \link array\endlink, and a pointer is NULL if the chunk has all zero elements; non-NULL pointer points to a chunk
	that has allocated memory and which stores elements of the vector.
*/
class budgetedVector
{
	/*! \var static unsigned int dimension
		\brief Dimensionality of the vector.
	*/
	/*! \var static unsigned int id
		\brief ID of the vector.

		Each vector is uniquely identifiable using its ID. This is used in AMM batch algorithm, where weights and data points are matched, and we need to know which weight
		(represented as \link budgetedVector \endlink), is assigned to which data point during stochastic gradient descent training.
	*/
	/*! \var static unsigned int arrayLength
		\brief Number of vector chunks.

		In order to deal with high-dimensional data, each vector is split into several chunks, and the memory for the chunk is not allocated if all elements of a vector are equal to 0.
		The static variable \link chunkWeight\endlink specifies how many of these chunks are used to represent each vector. \sa parameters::CHUNK_WEIGHT
	*/
	/*! \var static unsigned int chunkWeight
		\brief Length of the vector chunk (implemented as an array). \sa parameters::CHUNK_WEIGHT
	*/
	/*! \var vector <float*> array
		\brief Array of vector chunks, element of the array is NULL if all features within a chunk represented by the element are equal to 0.

		When the data is sparse, then we do not have to explicitly store every feature as most of them are equal to 0. One option is simply to follow LIBSVM format, and
		store in two linked lists feature index and the corresponing feature value. However, we found that updating this data structure can become prohibitively slow, as for
		high-dimensional data the weights can become much less sparse than the original data due to the weight update process, and the insertion of new elements into vector and vector traversal becomes very slow.
		We address this by storing a vector into structure that is a vector of dynamic arrays, where original, large vector is split into parts (or chunks), and each part is stored in an
		array within the vector structure. If all elements of the large vector within a chunk are zero, we do not allocate memory for that chunk and \link array\endlink element for this chunk will
		be NULL. In our experience, this significantly improves the training and testing time on very high-dimensional sparse data, such as on URL data set with more than
		3.2 million features and only 0.004% non-zero values. If \link parameters::CHUNK_WEIGHT \endlink is set to 1, we obtain the LIBSVM-type representation where each chunk
		stores only one feature. \sa parameters::CHUNK_WEIGHT
	*/
	/*! \var unsigned int weightID;
		\brief Unique ID of the vector, used in AMM batch to uniquely identify which vector is assigned to which data points. Assigned when the vector is created. \sa id
	*/
	/*! \var long double sqrL2norm;
		\brief Squared L2-norm of the vector.

		After every modification to a budgetedVector object (e.g., due to an update in Stochastic Gradient Descent (SGD) learning step of AMM or BSGD algorithms), this property is updated to reflect the current squared norm of the vector. This is done to speed up computations of kernel functions, as Gaussian kernel used in BSGD and LLSVM is computed much faster when we know squared norms of two vectors that are inputs to a kernel function. Also, in AMM it is used in pruning phase to find the weights that need to be deleted, as we will prune only weights that have small L2-norm.
	*/

	protected:
		static unsigned int dimension;
		static unsigned int id;
		static unsigned int arrayLength;
		static unsigned int chunkWeight;
        unsigned int weightID;
    	vector <float*> array;
		long double sqrL2norm;

		/*! \fn virtual void setSqrL2norm(double newSqrNorm)
			\brief Returns \link sqrL2norm\endlink, a squared L2-norm of the vector.
			\return Squared L2-norm of the vector.
		*/
		virtual void setSqrL2norm(long double newSqrNorm)
		{
			sqrL2norm = newSqrNorm;
		}

	public:
		/*! \fn virtual long double getSqrL2norm(void)
			\brief Returns \link sqrL2norm\endlink, a squared L2-norm of the vector.
			\return Squared L2-norm of the vector.
		*/
		virtual long double getSqrL2norm(void)
		{
			return sqrL2norm;
		}

		/*! \fn unsigned int getID(void)
			\brief Returns \link weightID\endlink, a unique ID of a vector.
			\return Unique ID of a vector.
		*/
		unsigned int getID(void)
		{
			return weightID;
		}


        /*! \fn const float operator[](int idx) const
			\brief Overloaded [] operator that returns a vector element stored in \link array\endlink.
			\param [in] idx Index of vector element that is retrieved.
			\return Value of the element of the vector.
		*/
		const float operator[](int idx) const;


		/*! \fn float& operator[](int idx)
			\brief Overloaded [] operator that assigns a value to vector element stored in \link array\endlink.
			\param [in] idx Index of vector element that is modified.
			\return Value of the modified element of the vector.
		*/
		float& operator[](int idx);

    	/*! \fn budgetedVector(long dim = 0, long chnkWght = 0)
			\brief Constructor, initializes the vector to all zeros.
			\param [in] dim Dimensionality of the vector.
			\param [in] chnkWght Size of each vector chunk.
		*/
		budgetedVector(unsigned int dim = 0, unsigned int chnkWght = 0)
		{
			if (dimension == 0)
				dimension = dim;
			if (chunkWeight == 0)
				chunkWeight = chnkWght;

			if (arrayLength == 0)
				arrayLength = (unsigned int)((dim - 1) / chunkWeight) + 1;

			// just initialize the elements of array to NULL, will be created only
			//      when needed, when one of the elements becomes non-zero
			for (unsigned int i = 0; i < arrayLength; i++)
				array.push_back(NULL);

			weightID = id++;
			sqrL2norm = 0.0;
		}

    	/*! \fn virtual ~budgetedVector()
			\brief Destructor, cleans up the memory.
		*/
		virtual ~budgetedVector()
		{
			this->clear();
			array.clear();
		}

		/*! \fn virtual void clear(void)
			\brief Clears the vector of all non-zero elements, resulting in a zero-vector.
		*/
		virtual void clear(void)
		{
			for (unsigned int i = 0; i < arrayLength; i++)
			{
				if (array[i] != NULL)
				{
					delete [] array[i];
					array[i] = NULL;
				}
			}
		}

		/*! \fn virtual void createVectorUsingDataPoint(budgetedData* inputData, unsigned int t, parameters* param)
			\brief Create new vector from training data point.
			\param [in] inputData Input data from which t-th vector is considered.
			\param [in] t Index of the input vector in the input data.
			\param [in] param The parameters of the algorithm.

			Initializes elements of a vector using a data point. Simply copies non-zero elements of the data point stored in budgetedData to the vector. If the vector already had non-zero elements, it is first cleared to become a zero-vector before copying the elements of a data point.
		*/
		virtual void createVectorUsingDataPoint(budgetedData* inputData, unsigned int t, parameters* param)
		{
			unsigned int ibegin = inputData->ai[t];
			unsigned int iend = (t == (unsigned int) (inputData->ai.size() - 1)) ? (unsigned int) (inputData->aj.size()) : inputData->ai[t + 1];

			this->clear();
			for (unsigned int i = ibegin; i < iend; i++)
			{
				((*this)[inputData->aj[i] - 1]) = inputData->an[i];
				sqrL2norm += (inputData->an[i] * inputData->an[i]);
			}
			if ((*param).BIAS_TERM != 0)
			{
				((*this)[(*param).DIMENSION - 1]) = (float)((long double)(*param).BIAS_TERM);
				sqrL2norm += ((*param).BIAS_TERM * (*param).BIAS_TERM);
			}
		};

        /*! \fn virtual long double sqrNorm(void)
			\brief Calculates a squared L2-norm of the vector.
			\return Squared L2-norm of the vector.
		*/
		virtual long double sqrNorm(void);

		/*! \fn virtual long double gaussianKernel(budgetedVector* otherVector, parameters *param)
			\brief Computes Gaussian kernel kernel between this budgetedVector vector and another vector stored in budgetedVector.
			\param [in] otherVector The second input vector to RBF kernel.
			\param [in] param The parameters of the algorithm.
			\return Value of RBF kernel between two vectors.

			Function computes the value of Gaussian kernel between two vectors. The computation is very fast for sparse data, being only linear in a number of non-zero features. We use the fact that ||x - y||^2 = ||x||^2 - 2 * x^T * y + ||y||^2, where all right-hand side elements can be computed efficiently.
		*/
		virtual long double gaussianKernel(budgetedVector* otherVector, parameters *param);

		/*! \fn virtual long double gaussianKernel(unsigned int t, budgetedData* inputData, parameters *param, long double inputVectorSqrNorm)
			\brief Computes Gaussian kernel kernel between this budgetedVector vector and another vector from input data stored in budgetedData.
			\param [in] t Index of the input vector in the input data.
			\param [in] inputData Input data from which t-th vector is considered.
			\param [in] inputVectorSqrNorm If zero or not provided, the norm of t-th vector from inputData is computed on-the-fly.
			\param [in] param The parameters of the algorithm.
			\return Value of RBF kernel between two vectors.

			Function computes the value of Gaussian kernel between two vectors. The computation is very fast for sparse data, being only linear in a number of non-zero features. We use the fact that ||x - y||^2 = ||x||^2 - 2 * x^T * y + ||y||^2, where all right-hand side elements can be computed efficiently.
		*/
		virtual long double gaussianKernel(unsigned int t, budgetedData* inputData, parameters *param, long double inputVectorSqrNorm = 0.0);

		/*! \fn virtual long double linearKernel(unsigned int t, budgetedData* inputData, parameters *param)
			\brief Computes linear kernel between this budgetedVector vector and another vector stored in budgetedData.
			\param [in] t Index of the input vector in the input data.
			\param [in] inputData Input data from which t-th vector is considered.
			\param [in] param The parameters of the algorithm.
			\return Value of linear kernel between two input vectors.

			Function computes the dot product of budgetedVector vector, and the input data point stored in budgetedData.
		*/
		virtual long double linearKernel(unsigned int t, budgetedData* inputData, parameters *param);

		/*! \fn virtual long double linearKernel(budgetedVector* otherVector)
			\brief Computes linear kernel between this budgetedVector vector and another vector stored in budgetedVector.
			\param [in] otherVector The second input vector to linear kernel.
			\return Value of linear kernel between two input vectors.

			Function computes the value of linear kernel between two vectors.
		*/
		virtual long double linearKernel(budgetedVector* otherVector);
};

/*! \class budgetedModel
    \brief Interface which defines methods to load model from and save model to text file.

	In order to ensure that all algorithms have the same interface when it comes to storing/loading of the trained model, this interface is to be implemented by each separate algorithm model.
*/
class budgetedModel
{
	public:
		/*! \fn static int getAlgorithm(const char *filename)
			\brief Get algorithm code from the trained model stored in .txt file, according to enumeration explained at the top of this page.
			\param [in] filename Filename of the .txt file where the model is saved.
			\return -1 if error, otherwise returns algorithm code from the model file.
		*/
		static int getAlgorithm(const char *filename);

		/*! \fn virtual ~budgetedModel()
			\brief Destructor, cleans up the memory.
		*/
		virtual ~budgetedModel(void) {};

		/*! \fn virtual bool saveToTextFile(const char *filename, vector <int>* yLabels, parameters *param) = 0
			\brief Saves the trained model to .txt file.
			\param [in] filename Filename of the .txt file where the model is saved.
			\param [in] yLabels Vector of possible labels.
			\param [in] param The parameters of the algorithm.
			\return False if error encountered, otherwise true.

			The text file has the following rows: [\a ALGORITHM, \a DIMENSION, \a NUMBER_OF_CLASSES, \a LABELS, \a NUMBER_OF_WEIGHTS, \a BIAS_TERM, \a KERNEL_WIDTH, \a MODEL]. In order to compress memory and to use the
			memory efficiently, we coded the model in the following way:

			- For AMM batch, AMM online, Pegasos:	The model is stored so that each row of the text file corresponds to one weight.
			The first element of each weight is the class of the weight, followed by the degradation of the weight. The rest of the row corresponds to non-zero elements of the weight, given
			as \a feature_index:feature_value, in a standard LIBSVM format.

			- For BSGD: The model is stored so that each row corresponds to one support vector (or weight). The first elements of each weight correspond to alpha parameters for each class,
			given in order by \a LABELS row. However, since alpha can be equal to 0, we use LIBSVM format to store alphas, as -\a class_index:class-\a specific_alpha, where we
			added '-' (minus sign) in front of the class index to differentiate between class indices and feature indices that follow. After the alphas, in the same row the elements of the
			weights (or support vectors) for each feature are given in LIBSVM format.

			- For LLSVM: The model is stored so that each row corresponds to one landmark point. The first element of each row corresponds to element of linear SVM hyperplane for that particular
			landmark point. This is followed by features of the landmark point in the original feature space of the data set in LIBSVM format.
		*/
		virtual bool saveToTextFile(const char *filename, vector <int>* yLabels, parameters *param) = 0;

		/*! \fn virtual bool loadFromTextFile(const char *filename, vector <int>* yLabels, parameters *param) = 0
			\brief Loads the trained model from .txt file.
			\param [in] filename Filename of the .txt file where the model is saved.
			\param [out] yLabels Vector of possible labels.
			\param [out] param The parameters of the algorithm.
			\return False if error encountered, otherwise true.

			The text file has the following rows: [\a ALGORITHM, \a DIMENSION, \a NUMBER_OF_CLASSES, \a LABELS, \a NUMBER_OF_WEIGHTS, \a BIAS_TERM, \a KERNEL_WIDTH, \a MODEL]. In order to compress memory and to use the
			memory efficiently, we coded the model in the following way:

			- For AMM batch, AMM online, Pegasos: The model is stored so that each row of the text file corresponds to one weight.
			The first element of each weight is the class of the weight, followed by the degradation of the weight. The rest of the row corresponds to non-zero elements of the weight, given
			as \a feature_index:feature_value, in a standard LIBSVM format.

			- For BSGD: The model is stored so that each row corresponds to one support vector (or weight). The first elements of each weight correspond to alpha parameters for each class,
			given in order specified by \a LABELS row. However, since alpha can be equal to 0, we use LIBSVM format to store alphas, as  -\a class_index:class-\a specific_alpha, where we
			added '-' (minus sign) in front of the class index to differentiate between class indices and feature indices that follow. After the alphas, in the same row the elements of the
			weights (or support vectors) for each feature are given in LIBSVM format.

			- For LLSVM: The model is stored so that each row corresponds to one landmark point. The first element of each row corresponds to element of linear SVM hyperplane for that particular
			landmark point. This is followed by features of the landmark point in the original feature space of the data set in LIBSVM format.
		*/
		virtual bool loadFromTextFile(const char *filename, vector <int>* yLabels, parameters *param) = 0;
};

/*! \fn void parseInputPrompt(int argc, char **argv, bool trainingPhase, char *inputFile, char *modelFile, char *outputFile, parameters *param)
	\brief Parses the user input from command prompt and modifies parameter settings as necessary, taken from LIBLINEAR implementation.
	\param [in] argc Argument count.
	\param [in] argv Argument vector.
	\param [in] trainingPhase True for training phase parsing, false for testing phase.
	\param [out] inputFile Filename of input data file.
	\param [out] modelFile Filename of model file.
	\param [out] outputFile Filename of output file (only used during testing phase).
	\param [out] param Parameter object modified by user input.
    \param [out] randomSeed Random seed for srand (optional)
*/
void parseInputPrompt(int argc, char **argv, bool trainingPhase, char *inputFile, char *modelFile, char *outputFile, parameters *param, int *randomSeed=NULL);

/*! \fn void printUsagePrompt(bool trainingPhase, parameters *param)
	\brief Prints the instructions on how to use the software to standard output.
	\param [in] trainingPhase Indicator if training or testing phase instructions.
	\param [in] param Parameter object modified by user input.
*/
void printUsagePrompt(bool trainingPhase, parameters *param);

#ifdef __cplusplus
}
#endif

#endif /* _BUDGETEDSVM_H */
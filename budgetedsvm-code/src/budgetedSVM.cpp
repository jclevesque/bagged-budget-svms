/*
	\file budgetedSVM.cpp
	\brief Implementation of classes used throughout the budgetedSVM toolbox.
*/
/*
	* This program is a free software; you can redistribute it and/or modify
	* it under the terms of the GNU General Public License as published by
	* the Free Software Foundation; either version 3 of the License, or
	* (at your option) any later version.

	Authors	:	Nemanja Djuric
	Name	:	budgetedSVM.cpp
	Date	:	November 29th, 2012
	Desc.	:	Implementation of classes used throughout the budgetedSVM toolbox.
	Version	:	v1.01
*/

#include <vector>
#include <sstream>
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <cstdlib>
#include <cmath>
using namespace std;

#include "budgetedSVM.h"

unsigned int budgetedVector::dimension = 0;
unsigned int budgetedVector::id = 0;
unsigned int budgetedVector::arrayLength = 0;
unsigned int budgetedVector::chunkWeight = 0;

/* \fn bool fgetWord(FILE *fHandle, char *str);
	\brief Reads one word string from an input file.
	\param [in] fHandle Handle to an open file from which one word is read.
	\param [out] str A character string that will hold the read word.
	\return True if end-of-line or end-of-file encountered after reading a word string, otherwise false.

	The function is similar to C++ functions fgetc() and getline(), only that it reads a single word from a file. A word is defined as a sequence of characters that does not contain a white-space character or new-line character '\n'. As a model in BudgetedSVM is stored in a text file where each line may corresponds to a single support vector, it is also useful to know if we reached the end of the line or the end of the file, which is indicated by the return value of the function.
*/
bool fgetWord(FILE *fHandle, char *str)
{
	char temp;
	unsigned char index = 0;
	bool wordStarted = false;
	while (1)//for (int i = 0; i < 20; i++)
	{
		temp = (char) fgetc(fHandle);

		if (temp == EOF)
		{
			str[index++] = '\0';
			return true;
		}

		switch (temp)
		{
			case ' ':
				if (wordStarted)
				{
					str[index++] = '\0';
					return false;
				}
				break;

			case '\n':
				str[index++] = '\0';
				return true;
				break;

			default:
				wordStarted = true;
				str[index++] = temp;
				break;
		}
	}
}

/* \fn static void printNull(const char *s)
	\brief Delibarately empty print function, used to turn off printing.
	\param [in] text Text to be (not) printed.
*/
static void printNull(const char *text)
{
	// deliberately empty
}

/* \fn static void printNull(const char *s)
	\brief Default error print function.
	\param [in] text Text to be printed.
*/
static void printErrorDefault(const char *text)
{
	// the function prints an error message and quits the program
	fputs(text, stderr);
	fflush(stderr);
	exit(1);
}

/* \fn static void printNull(const char *s)
	\brief Default print function.
	\param [in] text Text to be printed.
*/
static void printStringStdoutDefault(const char *text)
{
	fputs(text, stdout);
	fflush(stdout);
}
static funcPtr svmPrintStringStatic = &printStringStdoutDefault;
static funcPtr svmPrintErrorStringStatic = &printErrorDefault;

/* \fn void svmPrintString(const char* text)
	\brief Prints string to the output.
	\param [in] text Text to be printed.

	Prints string to the output. Exactly to which output should be specified by \link setPrintStringFunction\endlink, which modifies the callback that is invoked for printing. This is convinient when simple printf() can not be used, for example if we want to print to Matlab prompt. For example on how to set the printing function in Matlab environment, see the implementation of \link parseInputMatlab\endlink.
*/
void svmPrintString(const char* text)
{
	svmPrintStringStatic(text);
}

/* \fn void setPrintStringFunction(funcPtr printFunc)
	\brief Modifies a callback that prints a string.
	\param [in] printFunc New text-printing function.

	This function is used to modify the function that is used to print to standard output.
	After calling this function, which modifies the callback function for printing, the text is printed simply by invoking \link svmPrintString\endlink. \sa funcPtr
*/
void setPrintStringFunction(funcPtr printFunc)
{
	if (printFunc == NULL)
		svmPrintStringStatic = &printNull;
	else
		svmPrintStringStatic = printFunc;
}

/* \fn void svmPrintErrorString(const char* text)
	\brief Prints error string to the output.
	\param [in] text Text to be printed.

	Prints error string to the output. Exactly to which output should be specified by \link setPrintErrorStringFunction\endlink, which modifies the callback that is invoked for printing. This is convinient when an error is detected and, prior to printing appropriate message to a user, we want to exit the program. For example on how to set the printing function in Matlab environment, see the implementation of \link parseInputMatlab\endlink.
*/
void svmPrintErrorString(const char* text)
{
	svmPrintErrorStringStatic(text);
}

/* \fn void setPrintErrorStringFunction(funcPtr printFunc)
	\brief Modifies a callback that prints an error string.
	\param [in] printFunc New text-printing function.

	This function is used to modify the function that is used to print to error output.
	After calling this function, which modifies the callback function for printing error string, the text is printed simply by invoking \link svmPrintErrorString\endlink. \sa funcPtr
*/
void setPrintErrorStringFunction(funcPtr printFunc)
{
	if (printFunc == NULL)
		svmPrintErrorStringStatic = &printErrorDefault;
	else
		svmPrintErrorStringStatic = printFunc;
}

/* \fn bool readableFileExists(const char fileName[])
	\brief Checks if the file, identified by the input parameter, exists and is available for reading.
	\param [in] fileName Handle to an open file from which one word is read.
	\return True if the file exists and is available for reading, otherwise false.
*/
bool readableFileExists(const char fileName[])
{
	FILE *pFile = NULL;
	if (fileName)
	{
		pFile = fopen(fileName, "r");
		if (pFile != NULL)
		{
			fclose(pFile);
			return true;
		}
	}
	return false;
}

/* \fn static int getAlgorithm(const char *filename)
	\brief Get algorithm from the trained model stored in .txt file.
	\param [in] filename Filename of the .txt file where the model is saved.
	\return -1 if error, otherwise returns algorithm code from the model file.
*/
int budgetedModel::getAlgorithm(const char *filename)
{
	FILE *fModel = NULL;
	int temp;
	fModel = fopen(filename, "rt");
	if (!fModel)
		return -1;

	if (!fscanf(fModel, "ALGORITHM: %d\n", &temp))
	{
		svmPrintErrorString("Error reading algorithm type from the model file!\n");
	}
	return temp;
}

// vanilla initialization, just set everything to NULL
// if labels provided then initialize the labels array, used when testing
/* \fn budgetedData::budgetedData(bool keepAssignments, vector <int> *yLabels)
	\brief Vanilla constructor, just initializes the variables.
	\param [in] keepAssignments True for AMM batch, otherwise false.
	\param [in] yLabels Possible labels in the classification problem, for training data is NULL since inferred from data.
*/
budgetedData::budgetedData(bool keepAssignments, vector <int> *yLabels)
{
	this->ifileName = NULL;
	this->ifileNameAssign = NULL;
	this->ifile = NULL;
	this->assignments = NULL;
	this->al = NULL;
	this->keepAssignments = keepAssignments;
	this->loadTime = 0;
	this->N = 0;
	this->dataPartiallyLoaded = false;
	this->loadedDataPointsSoFar = 0;
	this->numNonZeroFeatures = 0;

	// if labels provided use them, this happens in the case of testing data
	if (yLabels)
	{
		for (unsigned int i = 0; i < (*yLabels).size(); i++)
		{
			this->yLabels.push_back((*yLabels)[i]);
		}
	}
}

/* \fn budgetedData(const char fileName[], unsigned int dimension, unsigned int chunkSize, bool keepAssignments = false, vector <int> *yLabels = NULL)
	\brief Constructor that takes the data from LIBSVM-style .txt file.
	\param [in] fileName Path to the input .txt file.
	\param [in] dimension Dimensionality of the classification problem.
	\param [in] chunkSize Size of the input data chunk that is loaded.
	\param [in] keepAssignments True for AMM batch, otherwise false.
	\param [in] yLabels Possible labels in the classification problem, for training data is NULL since inferred from data.
*/
budgetedData::budgetedData(const char fileName[], unsigned int dimension, unsigned int chunkSize, bool keepAssignments, vector <int> *yLabels)
{
	this->ifileName = strdup(fileName);
	this->dimension = dimension;
	this->dimensionHighestSeen = 0;

	this->al = new (nothrow) unsigned char[chunkSize];
	if (this->al == NULL)
	{
		svmPrintErrorString("Memory allocation error (budgetedData Constructor)!");
	}

	// keepAssignments is used for AMM_batch, where we hold the epoch assignments of data points to hyperplanes
	this->keepAssignments = keepAssignments;
	if (keepAssignments)
	{
		this->ifileNameAssign = strdup("temp_assigns.txt");
		this->assignments = new (nothrow) unsigned int[chunkSize];
	}
	else
		this->assignments = NULL;

	// if labels provided use them, this happens in the case of testing data
	if (yLabels)
	{
		for (unsigned int i = 0; i < (*yLabels).size(); i++)
		{
			this->yLabels.push_back((*yLabels)[i]);
		}
	}

	this->fileOpened = false;
	this->fileAssignOpened = false;
	this->loadTime = 0;
	this->N = 0;
	this->dataPartiallyLoaded = true;
	this->loadedDataPointsSoFar = 0;
	this->numNonZeroFeatures = 0;
}

/* \fn ~budgetedData(void)
	\brief Destructor, cleans up the memory.
*/
budgetedData::~budgetedData(void)
{
	delete [] this->al;
	flushData();

	if (this->assignments)
	{
		// delete all memory taken for keeping epoch assignments
		delete [] this->assignments;

		// if data is partially loaded (e.g., since it is too big to fit the memory), then also remove the file used to keep the assignments of the loaded data points
		if (dataPartiallyLoaded)
			remove(this->ifileNameAssign);
	}
}

/* \fn void saveAssignment(unsigned int *assigns)
	\brief Used for AMM batch to save current assignments.
	\param [in] assigns Current assignments.
*/
void budgetedData::saveAssignment(unsigned int *assigns)
{
	// no need for saving and loading to file, if data is fully (i.e., not partially) loaded, then everything is in the workspace (e.g., in the case of Matlab interface this can happen)
	if (!dataPartiallyLoaded)
	{
		if (assignments == NULL)
			assignments = new (nothrow) unsigned int[N];

		for (unsigned int i = 0; i < N; i++)
			*(assignments + i) = *(assigns + i);

		return;
	}

	fAssignFile = fopen(ifileNameAssign, "at");

	for (unsigned int i = 0; i < N; i++)
		fprintf(fAssignFile, "%d\n", *(assigns + i));

	fclose(fAssignFile);
};

/* \fn void readChunkAssignments(bool endOfFile)
	\brief Reads assignments for the current chunk.
	\param [in] endOfFile If the final chunk, close the assignment file.
*/
void budgetedData::readChunkAssignments(bool endOfFile)
{
	// if data is fully loaded from the beginning then just exit (e.g., can happen when BudgetedSVM is called from Matlab interface)
	if (!dataPartiallyLoaded)
		return;

	int tempInt;
	if (!fileAssignOpened)
	{
		fileAssignOpened = true;
		fAssignFile = fopen(ifileNameAssign, "rt");
	}

	for (unsigned int i = 0; i < N; i++)
	{
		// get the assignments (as opposed to initial iteration and reassignment phase
		// where we write the assignments, here we read them)
		if (!fscanf(fAssignFile, "%d\n", &tempInt))
		{
			svmPrintErrorString("Error reading assignments from the text file!\n");
		}
		*(assignments + i) = (unsigned int) tempInt;
	}

	if (endOfFile)
	{
		fileAssignOpened = false;
		fclose(fAssignFile);
	}
};

/* \fn void flushData(void)
	\brief Clears all data taken up by the current chunk.
*/
void budgetedData::flushData(void)
{
	ai.clear();
	aj.clear();
	an.clear();
	N = 0;
};

/* \fn bool readChunk(int size, bool assign = false)
	\brief Reads the next data chunk.
	\param [in] size Size of the chunk to be loaded.
	\param [in] assign True if assignment should be saved, false otherwise.
	\return True if just read the last data chunk, false otherwise.
*/
bool budgetedData::readChunk(unsigned int size, bool assign)
{
	string text;

	char line[65535];	// maximum length of the line to be read is set to 65535
	int pos, label;
	unsigned int counter = 0, dimSeen, pointIndex = 0;
	unsigned long start = clock();
	bool labelFound;

	// if not loaded from .txt file just exit
	if (!dataPartiallyLoaded)
		return false;

	flushData();
	if (!fileOpened)
	{
		this->ifile = fopen(ifileName, "rt");
		this->fileOpened = true;
		this->loadedDataPointsSoFar = 0;
		this->numNonZeroFeatures = 0;

		// if the very beginning, just create the assignment file if necessary
		if ((!assign) && (keepAssignments))
		{
			fAssignFile = fopen(ifileNameAssign, "wt");
			fclose(fAssignFile);
		}
	}

	// load chunk
	while (fgets(line, 65535, ifile))
	{
		N++;
		loadedDataPointsSoFar++;

		stringstream ss;
		ss << line;

		// get label
		if (ss >> text)
		{
            //skip lines starting with a # character.
            if (text.c_str()[0] == '#')
            {
                N--;
            	loadedDataPointsSoFar--;
                continue;
            }
			label = atoi(text.c_str());
			ai.push_back(pointIndex);

			// get yLabels, if label not seen before add it into the label array
			labelFound = false;
			for (unsigned int i = 0; i < yLabels.size(); i++)
			{
				if (yLabels[i] == label)
				{
					al[counter++] = (char) i;
					labelFound = true;
					break;
				}
			}
			if (!labelFound)
			{
				yLabels.push_back(label);
				al[counter++] = (char) (yLabels.size() - 1);
			}
		}

		// get feature values
		while (ss >> text)
		{
			if ((pos = (int) text.find(":")))
			{
				dimSeen = atoi(text.substr(0, pos).c_str());
				aj.push_back(dimSeen);
				an.push_back((float) atof(text.substr(pos + 1, text.length()).c_str()));
				pointIndex++;
				numNonZeroFeatures++;

				// if more features found than specified, print error message
				if (dimension < dimSeen)
				{
					sprintf(line, "Found more features than specified with '-d' option (specified: %d, found %d)!\nPlease check your settings.\n", dimension, dimSeen);
					svmPrintErrorString(line);
				}

				if (dimensionHighestSeen < dimSeen)
					dimensionHighestSeen = dimSeen;
			}
		}

		// check the size of chunk
		if (N == size)
		{
			// still data left to load, keep working
			loadTime += (clock() - start);
			return true;
		}
	}

	// got to the end of file, no more data left to load, exit nicely
	fclose(ifile);
	fileOpened = false;
	loadTime += (clock() - start);

	return false;
}

/* \fn float getElementOfVector(unsigned int vector, unsigned int element)
	\brief Returns an element of a vector stored in\link budgetedData\endlink structure.
	\param [in] vector Index of the vector (C-style indexing used, starting from 0; note that LibSVM format indices start from 1).
	\param [in] element Index of the element of the vector (C-style indexing used, starting from 0; note that LibSVM format indices start from 1).
	\return Element of the vector specified as an input.
*/
float budgetedData::getElementOfVector(unsigned int vector, unsigned int element)
{
	unsigned int maxPointIndex, pointIndexPointer;

	// check if vector index too big
	if (vector >= this->N)
	{
		svmPrintString("Warning: Vector index in getElementOfVector() function out of bounds, returning default value of 0.\n");
		return 0.0;
	}
	// check if element index too big
	if (element >= this->dimension)
	{
		svmPrintString("Warning: Element index in getElementOfVector() function out of bounds, returning default value of 0.\n");
		return 0.0;
	}

	pointIndexPointer = this->ai[vector];
	maxPointIndex = ((unsigned int)(vector + 1) == this->N) ? (unsigned int) (this->aj.size()) : this->ai[vector + 1];

	for (unsigned int i = pointIndexPointer; i < maxPointIndex; i++)
	{
		// if we found the element return its value
		if (this->aj[i] == element + 1)
			return this->an[i];

		// if we went over the index of the wanted element, then the element is equal to 0
		if (this->aj[i] > element + 1)
			return 0.0;
	}
	// if the wanted element is indexed higher than all non-zero elements, then it is equal to 0
	return 0.0;
}

/* \fn long double getVectorSqrL2Norm(unsigned int vector, parameters *param)
	\brief Returns a squared L2-norm of a vector stored in \link budgetedData\endlink structure.
	\param [in] vector Index of the vector (C-style indexing used, starting from 0; note that LibSVM format indices start from 1).
	\param [in] param The parameters of the algorithm.
	\return Squared L2-norm of a vector.

	This function returns squared L2-norm of a vector stored in the \link budgetedData\endlink structure. In particular, it is used to speed up the computation of Gaussian kernel.
*/
long double budgetedData::getVectorSqrL2Norm(unsigned int vector, parameters *param)
{
	unsigned int maxPointIndex, pointIndexPointer;
	long double result = 0.0;

	// check if vector index too big
	if (vector >= this->N)
	{
		svmPrintString("Warning: Vector index in getElementOfVector() function out of bounds, returning default value of 0.\n");
		return 0.0;
	}

	pointIndexPointer = this->ai[vector];
	maxPointIndex = ((unsigned int)(vector + 1) == this->N) ? (unsigned int)(this->aj.size()) : this->ai[vector + 1];

	for (unsigned int i = pointIndexPointer; i < maxPointIndex; i++)
		result += (this->an[i] * this->an[i]);
	if (param->BIAS_TERM != 0.0)
		result += (param->BIAS_TERM * param->BIAS_TERM);

	return result;
}

/* \fn double distanceBetweenTwoPoints(unsigned int index1, unsigned int index2)
	\brief Computes Euclidean distance between two data points from the input data.
	\param [in] index1 Index of the first data point.
	\param [in] index2 Index of the second data point.
	\return Euclidean distance between the two points.
*/
double budgetedData::distanceBetweenTwoPoints(unsigned int index1, unsigned int index2)
{
	// if distance to itself, return 0.0
	if (index1 == index2)
		return 0.0;

	long icurrent1 = ai[index1];
	long iend1 = (index1 == ai.size() - 1) ? aj.size() : ai[index1 + 1];
	long icurrent2 = ai[index2];
	long iend2 = (index2 == ai.size() - 1) ? aj.size() : ai[index2 + 1];
	double dotxx = 0.0, dotyy = 0.0, dotxy = 0.0;

	double currFeat1, currFeat2;
	while (1)
	{
		// traverse the vectors non-zero feature by non-zero feature
		if (icurrent1 < iend1)
			currFeat1 = (double) aj[icurrent1];
		else
			currFeat1 = INF;
		if (icurrent2 < iend2)
			currFeat2 = (double) aj[icurrent2];
		else
			currFeat2 = INF;

		if (currFeat1 == currFeat2)
		{
			dotxy += (an[icurrent1] * an[icurrent2]);
			dotxx += (an[icurrent1] * an[icurrent1]);
			dotyy += (an[icurrent2] * an[icurrent2]);

			icurrent1++;
			icurrent2++;
		}
		else
		{
			if (currFeat1 < currFeat2)
			{
				dotxx += (an[icurrent1] * an[icurrent1]);
				icurrent1++;
			}
			else
			{
				dotyy += (an[icurrent2] * an[icurrent2]);
				icurrent2++;
			}
		}

		if ((icurrent1 >= iend1) && (icurrent2 >= iend2))
			break;
	}
	return dotxx + dotyy - 2.0 * dotxy;
}

/* \fn const float operator[](int idx) const
	\brief Overloaded [] operator that returns value.
	\param [in] idx Index of vector element that is retrieved.
	\return Value of the element of the vector.
*/
const float budgetedVector::operator[](int idx) const
{
	unsigned int vectorInd = (unsigned int) (idx / (int) chunkWeight);
	unsigned int arrayInd = (unsigned int) (idx % (int) chunkWeight);

	// this means that all elements of this chunk are 0
	if (array[vectorInd] == NULL)
		return 0.0;
	else
		return *(array[vectorInd] + arrayInd);
}

/* \fn float& operator[](int idx)
	\brief Overloaded [] operator that assigns value.
	\param [in] idx Index of vector element that is modified.
	\return Value of the modified element of the vector.
*/
float& budgetedVector::operator[](int idx)
{
	unsigned int vectorInd = (unsigned int)(idx / (int) chunkWeight);
	unsigned int arrayInd = (unsigned int) (idx % (int) chunkWeight);

	// if all elements were zero, then first create the array and only
	//    then return the reference
	if (array[vectorInd] == NULL)
	{
		float *tempArray = NULL;
		unsigned long arraySize = chunkWeight;

		// if the last chunk, then it might be smaller than the rest
		if (vectorInd == (arrayLength - 1))
		{
			if (dimension == chunkWeight)
				tempArray = new (nothrow) float[chunkWeight];
			else
			{
				arraySize = dimension % chunkWeight;
				tempArray = new (nothrow) float[arraySize];
			}
		}
		else
			tempArray = new (nothrow) float[chunkWeight];

		if (tempArray == NULL)
		{
			svmPrintErrorString("Memory allocation error (budgetedVector assignment)!");
		}

		// null the array
		for (unsigned int j = 0; j < arraySize; j++)
			*(tempArray + j) = 0;

		array[vectorInd] = tempArray;
	}

	return *(array[vectorInd] + arrayInd);
}

/* \fn long double budgetedVector::sqrNorm(void)
	\brief Calculates a squared norm of the vector.
	\return Squared norm of the vector.
*/
long double budgetedVector::sqrNorm(void)
{
	long double tempSum = 0.0;
	unsigned long chunkSize = chunkWeight;

	for (unsigned int i = 0; i < arrayLength; i++)
	{
		if (array[i] != NULL)
		{
			if ((i == (arrayLength - 1)) && (dimension != chunkWeight))
				chunkSize = dimension % chunkWeight;

			for (unsigned int j = 0; j < chunkSize; j++)
				tempSum += ((long double)array[i][j] * (long double)array[i][j]);
		}
	}
	return tempSum;
}

/* \fn long double budgetedVector::gaussianKernel(budgetedVector* otherVector, parameters *param)
	\brief Computes Gaussian kernel kernel between this and some other vector.
	\param [in] otherVector The second input vector to RBF kernel.
	\param [in] param The parameters of the algorithm.
	\return Value of RBF kernel between two vectors.

	Function computes the value of Gaussian kernel between two vectors. The computation is very fast for sparse data, being only linear in a number of non-zero features. We use the fact that ||x - y||^2 = ||x||^2 - 2 * x^T * y + ||y||^2, where all right-hand side elements can be computed efficiently.
*/
long double budgetedVector::gaussianKernel(budgetedVector* otherVector, parameters *param)
{
	return exp(-0.5L * (long double)((*param).GAMMA_PARAM) * (sqrL2norm + otherVector->getSqrL2norm() - 2.0L * this->linearKernel(otherVector)));
}

/* \fn virtual long double budgetedVector::gaussianKernel(unsigned int t, budgetedData* inputData, parameters *param, inputVectorSqrNorm)
	\brief Computes Gaussian kernel kernel between this and other vector from input data stored in \link budgetedData\endlink.
	\param [in] t Index of the input vector in the input data.
	\param [in] inputData Input data from which t-th vector is considered.
	\param [in] param The parameters of the algorithm.
	\param [in] inputVectorSqrNorm If equal to zero or not provided, the norm of the t-th vector from inputData is computed on-the-fly.
	\return Value of RBF kernel between two vectors.

	Function computes the value of Gaussian kernel between two vectors. The computation is very fast for sparse data, being only linear in a number of non-zero features. We use the fact that ||x - y||^2 = ||x||^2 - 2 * x^T * y + ||y||^2, where all right-hand side elements can be computed efficiently.
*/
long double budgetedVector::gaussianKernel(unsigned int t, budgetedData* inputData, parameters *param, long double inputVectorSqrNorm)
{
	if (inputVectorSqrNorm == 0.0)
		inputVectorSqrNorm = inputData->getVectorSqrL2Norm(t, param);
	return exp(-0.5L * (long double)((*param).GAMMA_PARAM) * (this->sqrL2norm + inputVectorSqrNorm - 2.0L * this->linearKernel(t, inputData, param)));
}

/* \fn long double budgetedVector::linearKernel(unsigned int t, budgetedData* trainData, parameters *param)
	\brief Computes linear kernel between vector and given input data point.
	\param [in] t Index of the input vector in the input data.
	\param [in] trainData Input data from which t-th vector is considered.
	\param [in] param The parameters of the algorithm.
	\return Value of linear kernel between two input vectors.

	Function computes the dot product of \link budgetedVectorAMM \endlink vector, and the input data point from \link budgetedData \endlink.
*/
long double budgetedVector::linearKernel(unsigned int t, budgetedData* trainData, parameters *param)
{
	long double result = 0.0;
	long unsigned int pointIndexPointer = trainData->ai[t];
	long unsigned int maxPointIndex = ((unsigned int)(t + 1) == trainData->N) ? trainData->aj.size() : trainData->ai[t + 1];

	unsigned int idx, vectorInd, arrayInd;
    for (long unsigned int i = pointIndexPointer; i < maxPointIndex; i++)
	{
		idx = trainData->aj[i] - 1;
		vectorInd = (int) (idx / chunkWeight);
		arrayInd = (int) (idx % chunkWeight);

		// this means that all elements of this chunk are 0
		if (array[vectorInd] == NULL)
			continue;
		else
			result += array[vectorInd][arrayInd] * trainData->an[i];
	}
	if ((*param).BIAS_TERM != 0)
	    result += (((*this)[(*param).DIMENSION - 1]) * (*param).BIAS_TERM);

	return result;
}

/* \fn virtual long double linearKernel(budgetedVector* otherVector)
	\brief Computes linear kernel between this budgetedVector vector and another vector stored in budgetedVector.
	\param [in] otherVector The second input vector to linear kernel.
	\return Value of linear kernel between two input vectors.

	Function computes the value of linear kernel between two vectors.
*/
long double budgetedVector::linearKernel(budgetedVector* otherVector)
{
	long double result = 0.0L;
	unsigned long chunkSize = chunkWeight;
	for (unsigned int i = 0; i < arrayLength; i++)
	{
		// if either of them is NULL, meaning all-zeros vector chunk, move on to the next chunk
		if ((this->array[i] == NULL) || (otherVector->array[i] == NULL))
			continue;

		// now we know that i-th vector chunks of both vectors have non-zero elements, go one by one and compute linear kernel
		if ((i == (arrayLength - 1)) && (dimension != chunkWeight))
			chunkSize = dimension % chunkWeight;
		for (unsigned int j = 0; j < chunkSize; j++)
		{
			result += this->array[i][j] * otherVector->array[i][j];
		}
	}
	return result;
}

/* \fn void printUsagePrompt(bool trainingPhase)
	\brief Prints the instructions on how to use the software to standard output.
	\param [in] trainingPhase Indicator if training or testing phase instructions.
*/
void printUsagePrompt(bool trainingPhase, parameters *param)
{
	char text[256];
	if (trainingPhase)
	{
		svmPrintString("\n Usage:\n");
		svmPrintString(" budgetedsvm-train [options] train_file [model_file]\n\n");
		svmPrintString(" Inputs:\n");
		svmPrintString(" options\t- parameters of the model\n");
		svmPrintString(" train_file\t- url of training file in LIBSVM format\n");
		svmPrintString(" model_file\t- file that will hold a learned model\n");
		svmPrintString(" --------------------------------------------\n");
		svmPrintString(" Options are specified in the following format:\n");
		svmPrintString(" '-OPTION1 VALUE1 -OPTION2 VALUE2 ...'\n\n");
		svmPrintString(" The following options are available (default values in parentheses):\n\n");
		svmPrintString(" d - dimensionality of the data (MUST be set by a user)\n");
		sprintf(text,  " A - algorithm, which budget-scale SVM solver to use (%d):\n", (*param).ALGORITHM);
		svmPrintString(text);
		svmPrintString("       0 - Pegasos\n");
		svmPrintString("       1 - AMM batch\n");
		svmPrintString("       2 - AMM online\n");
		svmPrintString("       3 - LLSVM\n");
		svmPrintString("       4 - BSGD\n\n");
		sprintf(text, " e - number of training epochs in AMM and BSGD (%d)\n", (*param).NUM_EPOCHS);
		svmPrintString(text);
		sprintf(text, " s - number of subepochs (in AMM batch, %d)\n", (*param).NUM_SUBEPOCHS);
		svmPrintString(text);
		sprintf(text, " b - bias term in AMM, if 0 no bias added (%.1f)\n", (*param).BIAS_TERM);
		svmPrintString(text);
		sprintf(text, " k - pruning frequency, after how many examples is pruning done in AMM (%d)\n", (*param).K_PARAM);
		svmPrintString(text);
		svmPrintString(" C - pruning aggresiveness, sets the pruning threshold in AMM, OR\n");
		sprintf(text,  "       linear-SVM regularization paramater C in LLSVM (%.5f)\n", (*param).C_PARAM);
		svmPrintString(text);

		sprintf(text, " l - limit on the number of weights per class in AMM (%d)\n", (*param).LIMIT_NUM_WEIGHTS_PER_CLASS);
		svmPrintString(text);
		sprintf(text, " L - learning parameter in AMM and BSGD (%.5f)\n", (*param).LAMBDA_PARAM);
		svmPrintString(text);
		sprintf(text, " G - kernel width exp(-.5*gamma*||x-y||^2) in BSGD and LLSVM (1/DIMENSIONALITY)\n");
		svmPrintString(text);
		sprintf(text, " B - total SV set budget in BSGD, OR number of landmark points in LLSVM (%d)\n", (*param).BUDGET_SIZE);
		svmPrintString(text);
		svmPrintString(" M - budget maintenance strategy in BSGD (0 - removal; 1 - merging), OR\n");
		sprintf(text,  "       landmark selection in LLSVM (0 - random; 1 - k-means; 2 - k-medoids) (%d)\n\n", (*param).MAINTENANCE_SAMPLING_STRATEGY);
		svmPrintString(text);

		svmPrintString(" z - training and test file are loaded in chunks so that the algorithms can\n");
		svmPrintString("       handle budget files on weaker computers; z specifies number of examples\n");
		sprintf(text,  "       loaded in a single chunk of data (%d)\n", (*param).CHUNK_SIZE);
		svmPrintString(text);
		svmPrintString(" w - model weights are split in chunks, so that the algorithm can handle\n");
		svmPrintString("       highly dimensional data on weaker computers; w specifies number of\n");
		sprintf(text,  "       dimensions stored in one chunk (%d)\n", (*param).CHUNK_WEIGHT);
		svmPrintString(text);
		svmPrintString(" S - if set to 1 data is assumed sparse, if 0 data assumed non-sparse, used to\n");
		svmPrintString("       speed up kernel computations (default is 1 when percentage of non-zero\n");
		svmPrintString("       features is less than 5%, and 0 when percentage is larger than 5%)\n");
		sprintf(text, " v - verbose output; 1 to show the algorithm steps, 0 for quiet mode (%d)\n", (*param).VERBOSE);
		svmPrintString(text);
        svmPrintString(" R - Random seed to use for random number generator.\n\n");
	}
	else
	{
		svmPrintString("\n Usage:\n");
		svmPrintString(" budgetedsvm-predict [options] test_file model_file output_file\n\n");
		svmPrintString(" Inputs:\n");
		svmPrintString(" options\t- parameters of the model\n");
		svmPrintString(" test_file\t- url of test file in LIBSVM format\n");
		svmPrintString(" model_file\t- file that holds a learned model\n");
		svmPrintString(" output_file\t- url of file where output will be written\n");
		svmPrintString(" --------------------------------------------\n");
		svmPrintString(" Options are specified in the following format:\n");
		svmPrintString(" '-OPTION1 VALUE1 -OPTION2 VALUE2 ...'\n\n");
		svmPrintString(" The following options are available (default values in parentheses):\n\n");

		svmPrintString(" z - the training and test file are loaded in chunks so that the algorithm can\n");
		svmPrintString("       handle budget files on weaker computers; z specifies number of examples\n");
		sprintf(text,  "       loaded in a single chunk of data (%d)\n", (*param).CHUNK_SIZE);
		svmPrintString(text);
		svmPrintString(" w - the model weight is split in parts, so that the algorithm can handle\n");
		svmPrintString("       highly dimensional data on weaker computers; w specifies number of\n");
		sprintf(text,  "       dimensions stored in one chunk (%d)\n", (*param).CHUNK_WEIGHT);
		svmPrintString(text);
		svmPrintString(" S - if set to 1 data is assumed sparse, if 0 data assumed non-sparse, used to\n");
		svmPrintString("       speed up kernel computations (default is 1 when percentage of non-zero\n");
		svmPrintString("       features is less than 5%, and 0 when percentage is larger than 5%)\n");
		sprintf(text, " v - verbose output; 1 to show algorithm steps, 0 for quiet mode (%d)\n\n", (*param).VERBOSE);
		svmPrintString(text);
	}
}

/* \fn void parseInputPrompt(int argc, char **argv, bool trainingPhase, char *inputFile, char *modelFile, char *outputFile, parameters *param)
	\brief Parses the user input from command prompt and modifies parameter settings as necessary, taken from LIBLINEAR implementation.
	\param [in] argc Argument count.
	\param [in] argv Argument vector.
	\param [in] trainingPhase True for training phase parsing, false for testing phase.
	\param [out] inputFile Filename of input data file.
	\param [out] modelFile Filename of model file.
	\param [out] outputFile Filename of output file (only used during testing phase).
	\param [out] param Parameter object modified by user input.
*/
void parseInputPrompt(int argc, char **argv, bool trainingPhase, char *inputFile, char *modelFile, char *outputFile, parameters *param, int *randomSeed)
{
	vector <char> option;
	vector <float> value;
	int i;
	FILE *pFile = NULL;
	char text[1024];

	// parse options
	for (i = 1; i < argc; i++)
	{
		if (argv[i][0] != '-')
			break;
		++i;
		option.push_back(argv[i - 1][1]);
		value.push_back((float) atof(argv[i]));
	}

	if (trainingPhase)
	{
		if (i >= argc)
		{
			svmPrintErrorString("Error, input format not recognized. Run 'budgetedsvm-train' for help.\n");
		}

		pFile = fopen(argv[i], "r");
		if (pFile == NULL)
		{
			sprintf(text, "Can't open input file %s!\n", argv[i]);
			svmPrintErrorString(text);
		}
		else
		{
			fclose(pFile);
			strcpy(inputFile, argv[i]);
		}

		// take model file if provided by a user
		if (i < argc - 1)
			strcpy(modelFile, argv[i + 1]);
		else
		{
			char *p = strrchr(argv[i], '/');
			if (p == NULL)
				p = argv[i];
			else
				++p;
			sprintf(modelFile, "%s.model", p);
		}

		// modify parameters
		for (unsigned int i = 0; i < option.size(); i++)
		{
			switch (option[i])
			{
				case 'A':
					(*param).ALGORITHM = (unsigned int) value[i];
					if ((*param).ALGORITHM > 4)
					{
						sprintf(text, "Input parameter '-A %d' out of bounds!\nRun 'budgetedsvm-train()' for help.\n", (*param).ALGORITHM);
						svmPrintErrorString(text);
					}
					break;
				case 'e':
					(*param).NUM_EPOCHS = (unsigned int) value[i];
					break;
				case 'd':
					(*param).DIMENSION = (unsigned int) value[i];
					break;
				case 's':
					(*param).NUM_SUBEPOCHS = (unsigned int) value[i];
					break;
				case 'k':
					(*param).K_PARAM = (unsigned int) value[i];
					break;
				case 'C':
					(*param).C_PARAM = (double) value[i];
					if ((*param).C_PARAM < 0.0)
					{
						sprintf(text, "Input parameter '-C' should be a positive real number!\nRun 'budgetedsvm-train()' for help.\n");
						svmPrintErrorString(text);
					}
					break;
				case 'L':
					(*param).LAMBDA_PARAM = (double) value[i];
					if ((*param).LAMBDA_PARAM < 0.0)
					{
						sprintf(text, "Input parameter '-L' should be a positive real number!\nRun 'budgetedsvm-train()' for help.\n");
						svmPrintErrorString(text);
					}
					break;

				case 'B':
					(*param).BUDGET_SIZE = (unsigned int) value[i];
					break;
				case 'G':
					(*param).GAMMA_PARAM = (double) value[i];
					if ((*param).GAMMA_PARAM < 0.0)
					{
						sprintf(text, "Input parameter '-G' should be a positive real number!\nRun 'budgetedsvm-train()' for help.\n");
						svmPrintErrorString(text);
					}
					break;
				case 'M':
					(*param).MAINTENANCE_SAMPLING_STRATEGY = (unsigned int) value[i];
					break;

				case 'b':
					(*param).BIAS_TERM = (double) value[i];
					break;
				case 'v':
					(*param).VERBOSE = (value[i] != 0);
					break;
				case 'l':
					(*param).LIMIT_NUM_WEIGHTS_PER_CLASS = (unsigned int) value[i];
					if ((*param).LIMIT_NUM_WEIGHTS_PER_CLASS < 1)
					{
						sprintf(text, "Input parameter '-l' should be a positive real number!\nRun 'budgetedsvm-train()' for help.\n");
						svmPrintErrorString(text);
					}
					break;

				case 'z':
					(*param).CHUNK_SIZE = (unsigned int) value[i];
					if ((*param).CHUNK_SIZE < 1)
					{
						sprintf(text, "Input parameter '-z' should be a positive real number!\nRun 'budgetedsvm-train()' for help.\n");
						svmPrintErrorString(text);
					}
					break;
				case 'w':
					(*param).CHUNK_WEIGHT = (unsigned int) value[i];
					if ((*param).CHUNK_WEIGHT < 1)
					{
						sprintf(text, "Input parameter '-w' should be a positive real number!\nRun 'budgetedsvm-train()' for help.\n");
						svmPrintErrorString(text);
					}
					break;
				case 'S':
					(*param).VERY_SPARSE_DATA = (unsigned int) (value[i] != 0);
					break;

                case 'R':
                    randomSeed[0] = (unsigned int) (value[i]);
                    break;

                case 'N':
                    (*param).NUM_ITERS = (unsigned int) (value[i]);
                    break;

				default:
					sprintf(text, "Error, unknown input parameter '-%c'!\nRun 'budgetedsvm-train' for help.\n", option[i]);
					svmPrintErrorString(text);
					break;
			}
		}

		// check the MAINTENANCE_SAMPLING_STRATEGY validity
		if ((*param).ALGORITHM == LLSVM)
		{
			if ((*param).MAINTENANCE_SAMPLING_STRATEGY > 2)
			{
				// 0 - random removal, 1 - k-means, 2 - k-medoids
				sprintf(text, "Error, unknown input parameter '-M %d'!\nRun 'budgetedsvm-train' for help.\n", (*param).MAINTENANCE_SAMPLING_STRATEGY);
				svmPrintErrorString(text);
			}
		}
		else if ((*param).ALGORITHM == BSGD)
		{
			if ((*param).MAINTENANCE_SAMPLING_STRATEGY > 1)
			{
				// 0 - smallest removal, 1 - merging
				sprintf(text, "Error, unknown input parameter '-M %d'!\nRun 'budgetedsvm-train' for help.\n", (*param).MAINTENANCE_SAMPLING_STRATEGY);
				svmPrintErrorString(text);
			}
		}

		// shut down printing to screen if user specified so
		if (!(*param).VERBOSE)
			setPrintStringFunction(NULL);

		// no bias term for LLSVM and BSGD functions
		if (((*param).ALGORITHM == LLSVM) || ((*param).ALGORITHM == BSGD))
			(*param).BIAS_TERM = 0.0;

		if ((*param).VERBOSE)
		{
			svmPrintString("\n*** Training started with the following parameters:\n");
			switch ((*param).ALGORITHM)
			{
				case PEGASOS:
					svmPrintString("Algorithm \t\t\t: Pegasos\n");
					break;
				case AMM_ONLINE:
					svmPrintString("Algorithm \t\t\t: AMM online\n");
					break;
				case AMM_BATCH:
					svmPrintString("Algorithm \t\t\t: AMM batch\n");
					break;
				case BSGD:
					svmPrintString("Algorithm \t\t\t: BSGD\n");
					break;
				case LLSVM:
					svmPrintString("Algorithm \t\t\t: LLSVM\n");
					break;
			}

			if (((*param).ALGORITHM == PEGASOS) || ((*param).ALGORITHM == AMM_BATCH) || ((*param).ALGORITHM == AMM_ONLINE))
			{
				sprintf(text, "Lambda parameter\t\t: %f\n", (*param).LAMBDA_PARAM);
				svmPrintString(text);
				sprintf(text, "Bias term \t\t\t: %f\n", (*param).BIAS_TERM);
				svmPrintString(text);
				if ((*param).ALGORITHM != PEGASOS)
				{
					sprintf(text, "Pruning frequency k \t\t: %d\n", (*param).K_PARAM);
					svmPrintString(text);
					sprintf(text, "Pruning parameter C \t\t: %f\n", (*param).C_PARAM);
					svmPrintString(text);
					sprintf(text, "Max num. of weights per class \t: %d\n", (*param).LIMIT_NUM_WEIGHTS_PER_CLASS);
					svmPrintString(text);
					sprintf(text, "Number of epochs \t\t: %d\n\n", (*param).NUM_EPOCHS);
					svmPrintString(text);
				}
				else
					svmPrintString("\n");
			}
			else if ((*param).ALGORITHM == BSGD)
			{
				sprintf(text, "Number of epochs \t\t: %d\n", (*param).NUM_EPOCHS);
				svmPrintString(text);
				if ((*param).MAINTENANCE_SAMPLING_STRATEGY == 0)
					svmPrintString("Maintenance strategy \t\t: 0 (smallest removal)\n");
				else
					svmPrintString("Maintenance strategy \t\t: 1 (merging)\n");
				sprintf(text, "Lambda parameter \t\t: %f\n", (*param).LAMBDA_PARAM);
				svmPrintString(text);
				if ((*param).GAMMA_PARAM != 0.0)
				{
					sprintf(text, "Gaussian kernel width \t\t: %f\n", (*param).GAMMA_PARAM);
					svmPrintString(text);
				}
				else
					svmPrintString("Gaussian kernel width \t\t: 1 / DIMENSIONALITY\n");
				sprintf(text, "Size of the budget \t\t: %d\n\n", (*param).BUDGET_SIZE);
				svmPrintString(text);
			}
			else if ((*param).ALGORITHM == LLSVM)
			{
				if ((*param).VERBOSE)
				{
					switch ((*param).MAINTENANCE_SAMPLING_STRATEGY)
					{
						case 0:
							svmPrintString("Landmark sampling \t\t: 0 (random sampling)\n");
							break;
						case 1:
							svmPrintString("Landmark sampling \t\t: 1 (k-means initialization)\n");
							break;
						case 2:
							svmPrintString("Landmark sampling \t\t: 1 (k-medoids initialization)\n");
							break;
					}
					sprintf(text, "Number of landmark points \t: %d\n", (*param).BUDGET_SIZE);
					svmPrintString(text);
					sprintf(text, "C regularization parameter \t: %f\n", (*param).C_PARAM);
					svmPrintString(text);
					if ((*param).GAMMA_PARAM != 0.0)
					{
						sprintf(text, "Gaussian kernel width \t\t: %f\n\n", (*param).GAMMA_PARAM);
						svmPrintString(text);
					}
					else
						svmPrintString("Gaussian kernel width \t\t: 1 / DIMENSIONALITY\n\n");
				}
			}
            if (randomSeed != NULL)
            {
                sprintf(text, "Random seed given\t\t : %i\n\n", randomSeed[0]);
                svmPrintString(text);
            }
		}

		// increase dimensionality if bias term included
		if ((*param).BIAS_TERM != 0.0)
			(*param).DIMENSION++;

		// set gamma to default value of inverse dimensionality if not specified by a user
		if ((*param).GAMMA_PARAM == 0.0)
			(*param).GAMMA_PARAM = 1.0 / (*param).DIMENSION;
	}
	else
	{
		if (i >= argc - 2)
		{
			svmPrintErrorString("Error, input format not recognized. Run 'budgetedsvm-predict' for help.\n");
		}

		pFile = fopen(argv[i], "r");
		if (pFile == NULL)
		{
			sprintf(text, "Can't open input file %s!\n", argv[i]);
			svmPrintErrorString(text);
		}
		else
		{
			fclose(pFile);
			strcpy(inputFile, argv[i]);
		}

		pFile = fopen(argv[i + 1], "r");
		if (pFile == NULL)
		{
			sprintf(text, "Can't open model file %s!\n", argv[i + 1]);
			svmPrintErrorString(text);
		}
		else
		{
			fclose(pFile);
			strcpy(modelFile, argv[i + 1]);
		}

		pFile = fopen(argv[i + 2], "w");
		if (pFile == NULL)
		{
			sprintf(text, "Can't create output file %s!\n", argv[i + 2]);
			svmPrintErrorString(text);
		}
		else
		{
			fclose(pFile);
			strcpy(outputFile, argv[i + 2]);
		}

		// modify parameters
		for (unsigned int i = 0; i < option.size(); i++)
		{
			switch (option[i])
			{
				case 'v':
					(*param).VERBOSE = (value[i] != 0);
					break;
				case 'z':
					(*param).CHUNK_SIZE = (unsigned int) value[i];
					if ((*param).CHUNK_SIZE < 1)
					{
						sprintf(text, "Input parameter '-z' should be a positive real number!\nRun 'budgetedsvm-train()' for help.\n");
						svmPrintErrorString(text);
					}
					break;
				case 'w':
					(*param).CHUNK_WEIGHT = (unsigned int) value[i];
					if ((*param).CHUNK_WEIGHT < 1)
					{
						sprintf(text, "Input parameter '-w' should be a positive real number!\nRun 'budgetedsvm-train()' for help.\n");
						svmPrintErrorString(text);
					}
					break;
				case 'S':
					(*param).VERY_SPARSE_DATA = (unsigned int) (value[i] != 0);
					break;

				default:
					sprintf(text, "Error, unknown input parameter '-%c'!\nRun 'budgetedsvm-predict' for help.\n", option[i]);
					svmPrintErrorString(text);
					break;
			}
		}
	}
}
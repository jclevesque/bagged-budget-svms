/*
	\file bsgd.cpp
	\brief Implementation of BSGD algorithm.

	Implements the classes and functions used to train and test the Budgeted Stochastic Gradient Descent (BSGD) model.
*/
/*
	* This program is a free software; you can redistribute it and/or modify
	* it under the terms of the GNU General Public License as published by
	* the Free Software Foundation; either version 3 of the License, or
	* (at your option) any later version.

	Author	:	Nemanja Djuric
	Name	:	bsgd.cpp
	Date	:	November 20th, 2012
	Desc.	:	Implements the classes and functions used to train and test the Budgeted Stochastic Gradient Descent (BSGD) model.
	Version	:	v1.01
*/

#include <vector>
#include <sstream>
#include <time.h>
#include <cmath>
#include <algorithm>
#include <stdio.h>
#include <string.h>
using namespace std;

#include "budgetedSVM.h"
#include "bsgd.h"

unsigned int budgetedVectorBSGD::numClasses = 0;

/* prototypes of functions used to find kMax in the case of merging budget maintenance strategy
long double evaluateMergingObjectiveFunc(long double a1, long double a2, long double k12, long double x);
long double goldenSectionSearch(long double k12, long double a1, long double a2, long double a, long double b, long double tolerance);
long double* computeKmax(vector <budgetedVectorBSGD*>* v, unsigned int numSVs, unsigned int merge1, parameters *param);*/

/* \fn ~budgetedModelBSGD(void)
	\brief Destructor, cleans up memory taken by BSGD.
*/
budgetedModelBSGD::~budgetedModelBSGD(void)
{
	if (modelBSGD)
	{
		for (unsigned int i = 0; i < (*modelBSGD).size(); i++)
			delete (*modelBSGD)[i];
		(*modelBSGD).clear();

		delete modelBSGD;
		modelBSGD = NULL;
	}
}

/* \fn bool budgetedModelBSGD::saveToTextFile(const char *filename, vector <int>* yLabels, parameters *param)
	\brief Saves the trained model to .txt file.
	\param [in] filename Filename of the .txt file where the model is saved.
	\param [in] yLabels Vector of possible labels.
	\param [in] param The parameters of the algorithm.
	\return Returns false if error encountered, otherwise true.
*/
bool budgetedModelBSGD::saveToTextFile(const char *filename, vector <int>* yLabels, parameters *param)
{
	unsigned int i, j;
	FILE *fModel = NULL;
	fModel = fopen(filename, "wt");
	bool tempBool;

	if (!fModel)
		return false;

	// algorithm
	fprintf(fModel, "ALGORITHM: %d\n", (*param).ALGORITHM);

	// dimension
	fprintf(fModel, "DIMENSION: %d\n", (*param).DIMENSION);

	// number of classes
	fprintf(fModel, "NUMBER_OF_CLASSES: %d\n", (int) (*yLabels).size());

	// labels
	fprintf(fModel, "LABELS:");
	for (i = 0; i < (*yLabels).size(); i++)
		fprintf(fModel, " %d", (*yLabels)[i]);
	fprintf(fModel, "\n");

	// number of weights
	fprintf(fModel, "NUMBER_OF_WEIGHTS: ");
	fprintf(fModel, "%d\n", (int) (*modelBSGD).size());

	// bias parameter
	fprintf(fModel, "BIAS_TERM: %f\n", (*param).BIAS_TERM);

	// kernel width (GAMMA) parameter
	fprintf(fModel, "KERNEL_WIDTH: %f\n", (*param).GAMMA_PARAM);

	// save the model
	fprintf(fModel, "MODEL:\n");
	for (i = 0; i < (*modelBSGD).size(); i++)
	//for (i = 0; i < 50; i++)
	{
		for (j = 0; j < (*yLabels).size(); j++)
		{
			// alphas have negative index to differentiate them from features
			if ((*((*modelBSGD)[i])).alphas[j] != 0.0)
				fprintf(fModel, "-%d:%2.10f ", j + 1, (double)(*((*modelBSGD)[i])).alphas[j]);
		}

		// this tempBool is used so that the line doesn't end with a white-space, it makes our life
		// easier when reading word-by-word from the model file using fgetWord(); we can, of course,
		// do without it, but to avoid unnecessary checks during loading of the model we do it here
		tempBool = true;
		for (j = 0; j < (*param).DIMENSION; j++)					// for every feature
		{
			if ((*((*modelBSGD)[i]))[j] != 0.0)
			{
				if (tempBool)
				{
					fprintf(fModel, "%d:%2.10f", j + 1, (*((*modelBSGD)[i]))[j]);
					tempBool = false;
				}
				else
					fprintf(fModel, " %d:%2.10f", j + 1, (*((*modelBSGD)[i]))[j]);
			}
		}
		fprintf(fModel, "\n");
	}

	fclose(fModel);
	return true;
}

/* \fn bool budgetedModelBSGD::loadFromTextFile(const char *filename, vector <int>* yLabels, parameters *param)
	\brief Loads the trained model from .txt file.
	\param [in] filename Filename of the .txt file where the model is saved.
	\param [out] yLabels Vector of possible labels.
	\param [out] param The parameters of the algorithm.
	\return Returns false if error encountered, otherwise true.
*/
bool budgetedModelBSGD::loadFromTextFile(const char *filename, vector <int>* yLabels, parameters *param)
{
	unsigned int i, numClasses;
	float tempFloat;
	string text;
	char oneWord[1024];
	int pos, tempInt;
	vector <unsigned int> numWeights;
	FILE *fModel = NULL;
	fModel = fopen(filename, "rt");
	bool doneReadingBool;
	long double sqrNorm;

	if (!fModel)
		return false;

	// algorithm
	fseek (fModel, strlen("ALGORITHM: "), SEEK_CUR);
	if (!fscanf(fModel, "%d\n", &((*param).ALGORITHM)))
	{
		svmPrintErrorString("Error reading algorithm type from the model file!\n");
	}

	// dimension
	fseek (fModel, strlen("DIMENSION: "), SEEK_CUR);
	if (!fscanf(fModel, "%d\n", &((*param).DIMENSION)))
	{
		svmPrintErrorString("Error reading dimensions from the model file!\n");
	}

	// number of classes
	fseek (fModel, strlen("NUMBER_OF_CLASSES: "), SEEK_CUR);
	if (!fscanf(fModel, "%d\n", &numClasses))
	{
		svmPrintErrorString("Error reading number of classes from the model file!\n");
	}

	// labels
	fseek (fModel, strlen("LABELS: "), SEEK_CUR);
	for (i = 0; i < numClasses; i++)
	{
		if (!fscanf(fModel, "%d ", &tempInt))
		{
			svmPrintErrorString("Error reading labels from the model file!\n");
		}
		(*yLabels).push_back(tempInt);
	}

	// number of weights
	fseek (fModel, strlen("NUMBER_OF_WEIGHTS: "), SEEK_CUR);
	if (!fscanf(fModel, "%d\n", &tempInt))
	{
		svmPrintErrorString("Error reading number of weight from the model file!\n");
	}
	numWeights.push_back(tempInt);

	// bias parameter
	fseek (fModel, strlen("BIAS_TERM: "), SEEK_CUR);
	if (!fscanf(fModel, "%f\n", &tempFloat))
	{
		svmPrintErrorString("Error reading bias term from the model file!\n");
	}
	(*param).BIAS_TERM = tempFloat;

	// kernel width (GAMMA) parameter
	fseek (fModel, strlen("KERNEL_WIDTH: "), SEEK_CUR);
	if (!fscanf(fModel, "%f\n", &tempFloat))
	{
		svmPrintErrorString("Error reading kernel width from the model file!\n");
	}
	(*param).GAMMA_PARAM = tempFloat;

	// load the model
	fseek (fModel, strlen("MODEL:\n") + 1, SEEK_CUR);

	for (i = 0; i < numWeights[0]; i++)							// for every weight
	{
		budgetedVectorBSGD *eNew = new budgetedVectorBSGD((*param).DIMENSION, (*param).CHUNK_WEIGHT, numClasses);
		sqrNorm = 0.0L;

		// get alphas and features
		doneReadingBool = false;
		while (!doneReadingBool)
		{
			doneReadingBool = fgetWord(fModel, oneWord);
			if (strlen(oneWord) == 0)
				continue;

			text = oneWord;
			if ((pos = (int) text.find(":")))
			{
				tempInt = atoi(text.substr(0, pos).c_str());
				tempFloat = (float) atof(text.substr(pos + 1, text.length()).c_str());

				// alphas have negative index, features have positive
				if (tempInt > 0)
				{
					(*eNew)[tempInt - 1] = tempFloat;
					sqrNorm += (long double)(tempFloat * tempFloat);
				}
				else
					(*eNew).alphas[- tempInt - 1] = tempFloat;
			}
		}
		eNew->setSqrL2norm(sqrNorm);

		(*modelBSGD).push_back(eNew);
		eNew = NULL;
	}

	fclose(fModel);
	return true;
}

/* \fn void budgetedVectorBSGD::updateSV(budgetedVectorBSGD* v, long double kMax)
	\brief Updates the vector to obtain a merged vector, used during merging budget maintenance.
	\param [in] v Vector that is merged with this vector.
	\param [in] kMax Parameter that specifies how to combine them, using the following expression: currentVector <- kMax * currentVector + (1 - kMax) * v.
*/
void budgetedVectorBSGD::updateSV(budgetedVectorBSGD* v, long double kMax)
{
	unsigned long chunkSize = chunkWeight;
	unsigned long i, j;
	long double linKern = this->linearKernel(v);

	for (i = 0; i < arrayLength; i++)
	{
		if (this->array[i] != NULL)
		{
			if ((*v).array[i] == NULL)
			{
				if ((i == (arrayLength - 1)) && (dimension != chunkWeight))
					chunkSize = dimension % chunkWeight;
				for (j = 0; j < chunkSize; j++)
					array[i][j] = (float)(kMax * (long double) this->array[i][j]);
			}
			else
			{
				if ((i == (arrayLength - 1)) && (dimension != chunkWeight))
					chunkSize = dimension % chunkWeight;
				for (j = 0; j < chunkSize; j++)
					this->array[i][j] = (float)(kMax * this->array[i][j] + (1.0 - kMax) * (*v).array[i][j]);
			}
		}
		else
		{
			if ((*v).array[i] != NULL)
			{
				if ((i == (arrayLength - 1)) && (dimension != chunkWeight))
					chunkSize = dimension % chunkWeight;

				float *tempArray = new (nothrow) float[chunkSize];
				if (tempArray == NULL)
				{
					svmPrintErrorString("Memory allocation error (budgetedVector assignment)!");
				}

				// copy the array
				for (j = 0; j < chunkSize; j++)
					*(tempArray + j) = (float)((1.0 - kMax) * (*v).array[i][j]);

				this->array[i] = tempArray;
				tempArray = NULL;
			}
		}
	}

	// we also update the squared norm of the merged vector
	this->sqrL2norm = kMax * kMax * (long double) (this->sqrL2norm) + (1.0L - kMax) * (1.0L - kMax) * v->sqrNorm() + 2.0L * kMax * (1.0L - kMax) * linKern;
}

/* \fn long double alphaNorm(void)
	\brief Computes a norm of alpha vector.
	\return Norm of the alpha vector.
*/
long double budgetedVectorBSGD::alphaNorm(void)
{
	long double tempSum = 0.0;
	for (unsigned long i = 0; i < alphas.size(); i++)
		tempSum += (alphas[i] * alphas[i]);
	return tempSum;
}

/*! \fn long double evaluateMergingObjectiveFunc(long double a1, long double a2, long double k12, long double x)
	\brief Find the current value of merging objective function.
	\param [in] a1 Alpha (class-specific) value of the first point.
	\param [in] a2 Alpha (class-specific) value of the second point.
	\param [in] k12 Kernel value between two points that are being merged.
	\param [in] x Current value of merging parameter h.
	\return Value of merging objective function.

	Evaluates the value of the merging objective function during the merging strategy of the budget maintenance process. Used by \link goldenSectionSearch \endlink function.
*/
long double evaluateMergingObjectiveFunc(long double a1, long double a2, long double k12, long double x)
{
    return (-1.0 * (a1 * pow(k12, (1.0 - x) * (1.0 - x)) + a2 *  pow(k12, x * x)));
}

/*! \fn long double goldenSectionSearch(long double k12, long double a1, long double a2, long double a, long double b, long double tolerance)
	\brief Find the parameter k that specifies the position of the merged data point (merged = k * point1 + (1 - k) * point2).
	\param [in] k12 Kernel value between two points that are being merged.
	\param [in] a1 Alpha (class-specific) value of the first point.
	\param [in] a2 Alpha (class-specific) value of the second point.
	\param [in] a Lower bound on the interval where we seek the solution.
	\param [in] b Upper bound on the interval where we seek the solution.
	\param [in] tolerance Tolerance of the solution.
	\return Value of parameter k.

	Used during the merging strategy of the budget maintenance process. Uses efficient golden search optimization to find value of k that minimizes the merging objective function. Used by \link computeKmax\endlink function.
*/
long double goldenSectionSearch(long double k12, long double a1, long double a2, long double a, long double b, long double tolerance)
{
    long double gamma, p, q, fp, fq;
    /*golden search part*/
    gamma = (sqrt(5.0) - 1.0) / 2.0;

    p = b - gamma * (b - a);
    q = a + gamma * (b - a);

    fp = evaluateMergingObjectiveFunc(a1, a2, k12, p);
    fq = evaluateMergingObjectiveFunc(a1, a2, k12, q);

    while ((b - a) >= 2.0 * tolerance)
    {
        if (fp <= fq)
        {
            b = q;
            q = p;
            fq = fp;

            p = b - gamma * (b - a);
            fp = evaluateMergingObjectiveFunc(a1, a2, k12, p);
        }
        else
        {
            a = p;
            p = q;
            fp = fq;

            q = a + gamma * (b - a);
            fq = evaluateMergingObjectiveFunc(a1, a2, k12, q);
        }
    }

    return ((a + b) / 2.0);
}

/*! \fn long double* computeKmax(vector <budgetedVectorBSGD*>* v, unsigned int merge1, parameters *param)
	\brief Find which two vectors to merge.
	\param [in] v Support vector set.
	\param [in] merge1 Index of the support vector that is being merged.
	\param [in] param The parameters of the algorithm.
	\return An array:
		1) k that maximizes merging objective function;
		2) kernel value between first point and the merged point;
		3) kernel value between second point and the merged point;
		4) index of the second support vector to be merged.

	Used during the merging strategy of the budget maintenance process. Given an existing support vector, finds which other support vector to merge with it to incur the smallest
	degradation of the model due to the merging loss.
*/
long double* computeKmax(vector <budgetedVectorBSGD*>* v, unsigned int merge1, parameters *param)
{
	long double k12, kMax, kZ1, kZ2, a1, a2, loss, lossMin, d, zAlpha; // ancillary vars
	long double kMaxRet = 0.0, kZret1 = 0.0, kZret2 = 0.0;	// return vars
	unsigned int merge2 = 0;	// return vars
	long double* returnValues = new long double[4];

	lossMin = INF;
	for (unsigned int i = 0; i < (*v).size(); i++)
	{
		if (i == merge1)
			continue;

		k12 = (*v)[merge1]->gaussianKernel((*v)[i], param);
		a1 = 0;
		a2 = 0;
		for (unsigned int k = 0; k < (*v)[i]->alphas.size(); k++)
		{
			d = (*v)[merge1]->alphas[k] + (*v)[i]->alphas[k];
			if (d == 0)
			{
				d = 0.0001;
			}
			a1 += (*v)[merge1]->alphas[k] / d;
			a2 += (*v)[i]->alphas[k] / d;
		}

		if (a1 * a2 > 0)
		{
			kMax = goldenSectionSearch(k12, a1, a2, 0.0, 1.0, 0.0001);
		}
		else if (a1 > 0)
		{
			kMax = goldenSectionSearch(k12, a1, a2, 1.0, 6.0, 0.0001);
		}
		else
		{
			kMax = goldenSectionSearch(k12, a1, a2, -5.0, 0.0, 0.0001);
		}

		kZ1 = (long double) pow(k12, (1 - kMax) * (1 - kMax));
		kZ2 = (long double) pow(k12, kMax * kMax);

		loss = 0.0;
		for (unsigned int k = 0; k < (*v)[i]->alphas.size(); k++)
		{
			zAlpha = (*v)[merge1]->alphas[k] * kZ1 + (*v)[i]->alphas[k] * kZ2;
			loss += pow((*v)[merge1]->alphas[k], 2) + pow((*v)[i]->alphas[k], 2) + 2.0L * k12 * (*v)[merge1]->alphas[k] * (*v)[i]->alphas[k] - zAlpha * zAlpha;
		}

		if (loss < lossMin)
		{
			lossMin = loss;
			kMaxRet = kMax;
			kZret1 = kZ1;
			kZret2 = kZ2;
			merge2 = i;
		}
	}

	if (kMaxRet < 0)
		kMaxRet = 0;

	returnValues[0] = kMaxRet;
	returnValues[1] = kZret1;
	returnValues[2] = kZret2;
	returnValues[3] = (long double) merge2;
	return returnValues;
}

/* \fn float predictBSGD(budgetedData *testData, parameters *param, budgetedModel *model, vector <char> *labels)
	\brief Given a BSGD model, predict the labels of testing data.
	\param [in] testData Input test data.
	\param [in] param The parameters of the algorithm.
	\param [in] model Trained BSGD model.
	\param [out] labels Vector of predicted labels.
	\return Testing set error rate.

	Given the learned BSGD model, the function computes the predictions on the testing data, outputing the predicted labels and the error rate.
*/
float predictBSGD(budgetedData *testData, parameters *param, budgetedModelBSGD *model, vector <char> *labels)
{
    unsigned long timeCalc = 0, start;
    unsigned int i, N, err = 0, total = 0;
	long double fx, maxFx, tempSqrNorm = 0.0, *tempArray;
	bool stillChunksLeft = true;
	char text[1024];
	int y;
	budgetedVectorBSGD *currentData = NULL;

	// this tempArray is used when calculating all class scores, to avoid repeated computations of the same kernel
	tempArray = new long double[(*(model->modelBSGD)).size()];
	for (i = 0; i < (*(model->modelBSGD)).size(); i++)
		tempArray[i] = 0.0;

	while (stillChunksLeft)
	{
        stillChunksLeft = testData->readChunk((*param).CHUNK_SIZE);
		(*param).updateVerySparseDataParameter(testData->getSparsity());

        N = testData->N;
        total += N;
		start = clock();
    	for (unsigned int r = 0; r < N; r++)
    	{
			if ((*param).VERY_SPARSE_DATA)
			{
				// since we are computing kernels using vectors directly from the budgetedData, we need square norm of the vector to speed-up
				// 	computations, here we compute it just once; no need to do it in non-sparse case, since this norm can be retrieved directly
				// 	from budgetedVector
				tempSqrNorm = testData->getVectorSqrL2Norm(r, param);
			}
			else
			{
				// create the budgetedVector using the vector from budgetedData, to be used in kernel computations below
				currentData = new budgetedVectorBSGD((*param).DIMENSION, (*param).CHUNK_WEIGHT, (unsigned int) (testData->yLabels).size());
				currentData->createVectorUsingDataPoint(testData, r, param);
			}

    		y = 0;
    		maxFx = -INF;
			for (i = 0; i < (*(model->modelBSGD)).size(); i++)
				tempArray[i] = 0.0;

    		for (unsigned int k = 0; k < (testData->yLabels).size(); k++)
    		{
				fx = 0;
				for (unsigned int i = 0; i < (*(model->modelBSGD)).size(); i++)
				{
					if ((*(model->modelBSGD))[i]->alphas[k] != 0)
					{
						if (tempArray[i] == 0.0)
						{
							if ((*param).VERY_SPARSE_DATA)
								// directly compute kernel from the trainData
								tempArray[i] = (*((*model).modelBSGD))[i]->gaussianKernel(r, testData, param, tempSqrNorm);
							else
								// compute kernel from currentData object
								tempArray[i] = (*(model->modelBSGD))[i]->gaussianKernel(currentData, param);
						}
						fx += ((*(model->modelBSGD))[i]->alphas[k] * tempArray[i]);
					}
				}

				if (fx > maxFx)
				{
					maxFx = fx;
					y = k;
				}
			}

			if (!(*param).VERY_SPARSE_DATA)
			{
				// if sparse then no need for this, since we didn't even create currentData
				delete currentData;
				currentData = NULL;
			}

    		if (y != testData->al[r])
    			err++;

			// save predicted label, will be sent to output
			if (labels)
				(*labels).push_back((char)(testData->yLabels)[y]);
    	}

		timeCalc += clock() - start;

		if (((*param).VERBOSE) && (N > 0))
    	{
			sprintf(text, "Number of examples processed: %d\n", total);
			svmPrintString(text);
        }
    }
	testData->flushData();
	delete [] tempArray;

	if ((*param).VERBOSE)
    {
		sprintf(text, "*** Testing completed in %5.3f seconds\n*** Testing error rate: %3.2f percent", (double)timeCalc / (double)CLOCKS_PER_SEC, 100.0 * (float)err / (float)total);
		svmPrintString(text);
    }

	return (float) (100.0 * (float)err / (float)total);
}

/* \fn void trainBSGD(budgetedData *trainData, parameters *param, budgetedModelBSGD *model)
	\brief Train BSGD.
	\param [in] trainData Input training data.
	\param [in] param The parameters of the algorithm.
	\param [in,out] model Initial BSGD model.

	The function trains BSGD model, given input data, the initial model (most often zero-weight model), and the parameters of the model.
*/
void trainBSGD(budgetedData *trainData, parameters *param, budgetedModelBSGD *model)
{
	unsigned long timeCalc = 0, start;
	long double fxValue, fxValue1, fxValue2, maxFx, *tempArray, alphaSmallest = 0.0, tempLongDouble = 0.0;
	unsigned int i1, i2 = 0, t, countDel = 0, numClasses = 0, numSVs = 0, numIter = 0, N, deleteWeight = 0;
	bool stillChunksLeft = true;
	char text[1024];
	unsigned int i, k, ot; 	//iterators
	budgetedVectorBSGD *currentData = NULL;

	// this tempArray is used when calculating all class scores and runner-up, to avoid repeated computations of the same kernel
	tempArray = new long double[(*param).BUDGET_SIZE];
	for (i = 0; i < (*param).BUDGET_SIZE; i++)
		tempArray[i] = 0.0;

	for (unsigned int epoch = 0; epoch < (*param).NUM_EPOCHS; epoch++)
	{
		//Calculate
		stillChunksLeft = true;
		while (stillChunksLeft)
		{
			stillChunksLeft = trainData->readChunk((*param).CHUNK_SIZE);
			(*param).updateVerySparseDataParameter(trainData->getSparsity());

			N = trainData->N;
			if (numIter == 0)
				numClasses = (unsigned int) trainData->yLabels.size();
			else if (numClasses != (unsigned int) trainData->yLabels.size())
			{
				// if in the earlier chunks some class wasn't observed, it could happen with small chunks or unbalanced classes;
				// just add new zero alphas for the new classes to each support vector
				for (unsigned int i = 0; i < numSVs; i++)
					for (unsigned int k = 0; k < (trainData->yLabels.size() - numClasses); k++)
						(*((*model).modelBSGD))[i]->alphas.push_back(0.0);
				numClasses = (unsigned int) trainData->yLabels.size();
			}

			// randomize
			vector <unsigned int> tv(N, 0);
			for (unsigned int ti = 0; ti < N; ti++)
			{
				tv[ti] = ti;
			}
			random_shuffle(tv.begin(), tv.end());

			start = clock();
			for (ot = 0; ot < N; ot++)
			{
				t = tv[ot];
				numIter++;
                if ((*param).NUM_ITERS > 0 and numIter > (*param).NUM_ITERS)
                {
                    stillChunksLeft = false;
                    printf("Stopping training at iteration %i\n", numIter);
                    break;
                }

				// initialize the first weight
				if (numIter == 1)
				{
					currentData = new budgetedVectorBSGD((*param).DIMENSION, (*param).CHUNK_WEIGHT, numClasses);
					currentData->createVectorUsingDataPoint(trainData, t, param);

					i1 = trainData->al[t];
					i2 = (i1 + 1) % numClasses;
					currentData->alphas[i1] = 1.0;
					currentData->alphas[i2] = -1.0;

					(*((*model).modelBSGD)).push_back(currentData);
					currentData = NULL;
					numSVs++;
					continue;
				}

				// calculate all class scores and runner-up
				i1 = trainData->al[t];
				fxValue1 = 0.0;
				fxValue2 = 0.0;
				maxFx = -INF;
				for (i = 0; i < numSVs; i++)
					tempArray[i] = 0.0;

				if ((*param).VERY_SPARSE_DATA)
				{
					// since we are computing kernels using vectors directly from the budgetedData, we need square norm of the vector to speed-up
					// 	computations, here we compute it just once; no need to do it in non-sparse case, since this norm can be retrieved directly
					// 	from budgetedVector
					tempLongDouble = trainData->getVectorSqrL2Norm(t, param);
				}
				else
				{
					// create the budgetedVector using the vector from budgetedData, to be used in gaussianKernel() method below
					currentData = new budgetedVectorBSGD((*param).DIMENSION, (*param).CHUNK_WEIGHT, numClasses);
					currentData->createVectorUsingDataPoint(trainData, t, param);
				}

				for (k = 0; k < numClasses; k++)
				{
					fxValue = 0.0;
					for (i = 0; i < numSVs; i++)
					{
						if ((*((*model).modelBSGD))[i]->alphas[k] != 0)
						{
							// calculate the kernel only if not computed earlier
							if (tempArray[i] == 0.0)
							{
								if ((*param).VERY_SPARSE_DATA)
									// directly compute kernel from the trainData
									tempArray[i] = (*((*model).modelBSGD))[i]->gaussianKernel(t, trainData, param, tempLongDouble);
								else
									// compute kernel from currentData object
									tempArray[i] = (*((*model).modelBSGD))[i]->gaussianKernel(currentData, param);
							}
							fxValue += ((*((*model).modelBSGD))[i]->alphas[k] * tempArray[i]);
						}
					}

					if (k == i1)
						fxValue1 = fxValue;
					else if (fxValue > maxFx)
					{
						maxFx = fxValue;
						i2 = k;
					}
				}
				fxValue2 = maxFx;

				// downweight all the weights
				for (i = 0; i < numSVs; i++)
					(*((*model).modelBSGD))[i]->downgrade(numIter);

				if (1.0 + fxValue2 - fxValue1 > 0.0)
				{
					if ((*param).VERY_SPARSE_DATA)
					{
						// only do this if data is sparse, since if non-sparse than we already have currentData initialized
						// 	from the code before the loop in which we computed kernels
						currentData = new budgetedVectorBSGD((*param).DIMENSION, (*param).CHUNK_WEIGHT, numClasses);
						currentData->createVectorUsingDataPoint(trainData, t, param);
					}

					// add an SV
					currentData->alphas[i1] =  1.0 / ((long double)numIter * (*param).LAMBDA_PARAM);
					currentData->alphas[i2] = -1.0 / ((long double)numIter * (*param).LAMBDA_PARAM);
					(*((*model).modelBSGD)).push_back(currentData);
					currentData = NULL;
					numSVs++;

					// if over the budget, maintain the budget
					if (numSVs > (*param).BUDGET_SIZE)
					{
						switch ((*param).MAINTENANCE_SAMPLING_STRATEGY)
						{
							case 0:
								// removal of an SV, remove the smallest one
								alphaSmallest = INF;
								for (i = 0; i < numSVs; i++)
								{
									tempLongDouble = (*((*model).modelBSGD))[i]->alphaNorm();
									if (alphaSmallest > tempLongDouble)
									{
										alphaSmallest = tempLongDouble;
										deleteWeight = i;
									}
								}

								delete (*((*model).modelBSGD))[deleteWeight];
								(*((*model).modelBSGD)).erase((*((*model).modelBSGD)).begin() + deleteWeight);
								break;

							case 1:
								// merging of two SVs
								long double kMax, kZ1, kZ2;
								unsigned int merge1, merge2;

								// find the one with smallest alpha
								// here we look for who to merge
								merge1 = 0;
								for (i = 0; i < numSVs; i++)
								{
									tempLongDouble = (*((*model).modelBSGD))[i]->alphaNorm();
									if ((alphaSmallest > tempLongDouble) || (i == 0))
									{
										alphaSmallest = tempLongDouble;
										merge1 = i;
									}
								}

								// find with who to merge, as well as other useful information detailed in the definition of computeKmax() found in this file
								long double* returnValues = computeKmax((*model).modelBSGD, merge1, param);
								kMax = (*returnValues);
								kZ1 = (*(returnValues + 1));
								kZ2 = (*(returnValues + 2));
								merge2 = (unsigned int) (*(returnValues + 3));
								delete [] returnValues;

								// find z, the new support vector
								(*((*model).modelBSGD))[merge1]->updateSV((*((*model).modelBSGD))[merge2], kMax);
								for (unsigned int k = 0; k < numClasses; k++)
									(*((*model).modelBSGD))[merge1]->alphas[k] = (*((*model).modelBSGD))[merge1]->alphas[k] * kZ1 + (*((*model).modelBSGD))[merge2]->alphas[k] * kZ2;

								// delete 'merge2', not needed anymore
								delete (*((*model).modelBSGD))[merge2];
								(*((*model).modelBSGD)).erase((*((*model).modelBSGD)).begin() + merge2);
								break;
						}
						numSVs--;
						countDel++;
					}
				}
				else
				{
					if (!(*param).VERY_SPARSE_DATA)
					{
						// if sparse data then no need for this part, since we didn't even create currentData
						delete currentData;
						currentData = NULL;
					}
				}
			}
    		timeCalc += clock() - start;

    		if (((*param).VERBOSE) && (N > 0))
			{
				sprintf(text, "Number of examples processed: %d\n", numIter);
				svmPrintString(text);
            }
		}

		if ((*param).VERBOSE && ((*param).NUM_EPOCHS > 1))
		{
			sprintf(text, "Epoch %d/%d done.\n", epoch + 1, (*param).NUM_EPOCHS);
			svmPrintString(text);
		}
	}
	delete [] tempArray;
	trainData->flushData();

	if ((*param).VERBOSE && ((*param).NUM_EPOCHS > 1))
		svmPrintString("\n");

	if ((*param).VERBOSE)
	{
		sprintf(text, "Training completed in %5.3f seconds.\n\nNumber of budget maintenance steps: %d\n", (double)timeCalc / (double)CLOCKS_PER_SEC, countDel);
		svmPrintString(text);
	}
}

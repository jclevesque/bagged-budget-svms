/*
	\file mm_algs.cpp
	\brief Implementation of AMM batch, AMM online and Pegasos algorithms.
	
	Implements classes and functions used for training and testing of large-scale multi-hyperplane algorithms (AMM batch, AMM online, and Pegasos).
*/
/* 
	* This program is a free software; you can redistribute it and/or modify
	* it under the terms of the GNU General Public License as published by
	* the Free Software Foundation; either version 3 of the License, or
	* (at your option) any later version.
	
	Author	:	Nemanja Djuric
	Name	:	mm_algs.cpp
	Date	:	November 19th, 2012
	Desc.	:	Source file for C++ implementation of budget-scale multi-hyperplane algorithms (AMM-batch, AMM-online, PEGASOS).
	Version	:	v1.01
*/

#include <cmath>
#include <vector>
#include <sstream>
#include <time.h>
#include <algorithm>
#include <stdio.h>
#include <string.h>
using namespace std;

#include "budgetedSVM.h"
#include "mm_algs.h"

/* \fn ~budgetedModelAMM(void)
	\brief Destructor, cleans up memory taken by AMM.
*/	
budgetedModelAMM::~budgetedModelAMM(void)
{
	if (modelMM)
	{
		for (unsigned int i = 0; i < (*modelMM).size(); i++)
		{           
			for (unsigned int j = 0; j < (*modelMM)[i].size(); j++)
				delete (*modelMM)[i][j];
			(*modelMM)[i].clear();
		}
		(*modelMM).clear();
		
		delete modelMM;
		modelMM = NULL;
	}
}

/* \fn bool budgetedModelAMM::saveToTextFile(const char *filename, vector <int>* yLabels, parameters *param)
	\brief Saves the trained AMM model to .txt file.
	\param [in] filename Filename of the .txt file where the model is saved.
	\param [in] yLabels Vector of possible labels.
	\param [in] param The parameters of the algorithm.
	\return Returns false if error encountered, otherwise true.
*/
bool budgetedModelAMM::saveToTextFile(const char *filename, vector <int>* yLabels, parameters *param)
{
	unsigned int i, j, k;
	FILE *fModel = NULL;
	fModel = fopen(filename, "wt");
	
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
	fprintf(fModel, "NUMBER_OF_WEIGHTS:");
	for (i = 0; i < (*modelMM).size(); i++)
		fprintf(fModel, " %d", (int) (*modelMM)[i].size());
	fprintf(fModel, "\n");
	
	// bias parameter
	fprintf(fModel, "BIAS_TERM: %f\n", (*param).BIAS_TERM);
	
	// kernel width (GAMMA) parameter
	fprintf(fModel, "KERNEL_WIDTH: 0.0\n");
	
	// save the model
	fprintf(fModel, "MODEL:\n");
	for (i = 0; i < yLabels->size(); i++)				// for every class
	{         
		for (j = 0; j < (*modelMM)[i].size(); j++)		// for every weight
		{
			// weight label
			fprintf(fModel, "%d ", (*yLabels)[i]);
			
			// degradation
			fprintf(fModel, "%2.10f", (double)((*modelMM)[i][j])->getDegradation());
			
			for (k = 0; k < (*param).DIMENSION; k++)	// for every feature
			{
				if ((*((*modelMM)[i][j]))[k] != 0.0)
					fprintf(fModel, " %d:%2.10f", k + 1, (*((*modelMM)[i][j]))[k]);
			}
			fprintf(fModel, "\n");
		}
	}
	
	fclose(fModel);
	return true;
}

/* \fn bool budgetedModelAMM::loadFromTextFile(const char *filename, vector <int>* yLabels, parameters *param)
	\brief Loads the trained AMM model from .txt file.
	\param [in] filename Filename of the .txt file where the model is saved.
	\param [out] yLabels Vector of possible labels.
	\param [out] param The parameters of the algorithm.
	\return Returns false if error encountered, otherwise true.
*/
bool budgetedModelAMM::loadFromTextFile(const char *filename, vector <int>* yLabels, parameters *param)
{
	unsigned int i, j, tempInt, numClasses;
	float tempFloat;
	string text;
	char oneWord[1024];
	int pos;
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
	for (i = 0; i < numClasses; i++)
	{
		if (!fscanf(fModel, "%d\n", &tempInt))
		{
			svmPrintErrorString("Error reading number of weights from the model file!\n");
		}
		numWeights.push_back(tempInt);
	}
	
	// bias parameter
	fseek(fModel, strlen("BIAS_TERM: "), SEEK_CUR);
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
	for (i = 0; i < numClasses; i++)								// for every class
	{
		// add for each class an empty weight matrix
		vector <budgetedVectorAMM*> tempV;
		(*modelMM).push_back(tempV);
		
		for (j = 0; j < numWeights[i]; j++)							// for every weight
		{			
			budgetedVectorAMM *eNew = new budgetedVectorAMM((*param).DIMENSION, (*param).CHUNK_WEIGHT);
			sqrNorm = 0.0L;
			
			// get degradation and features
			
			// skip label, no need to read it explicitly since we know the number of weights of each class, found in numWeights vector
			fgetWord(fModel, oneWord);
	
			// get degradation
			doneReadingBool = fgetWord(fModel, oneWord);
			eNew->setDegradation((long double) atof(oneWord));			
			
			// get features
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
					(*eNew)[tempInt - 1] = tempFloat;
					
					sqrNorm += (long double)(tempFloat * tempFloat);
				}
			}
			eNew->setSqrL2norm(sqrNorm);
			
			(*modelMM)[i].push_back(eNew);			
			eNew = NULL;
		}
	}
	
	fclose(fModel);
	return true;
}

/* \fn void budgetedVectorAMM::updateUsingDataPoint(budgetedData* inputData, unsigned int oto, unsigned int t, int sign, parameters *param)
	\brief Updates existing weight when misclassification happens.
	\param [in,out] vij Existing weight that needs to be updated.
	\param [in] inputData Input data from which t-th vector is considered.
	\param [in] oto Total number of iterations so far.
	\param [in] t Index of the input vector in the input data.
	\param [in] sign +1 if the input vector is of the true class, -1 otherwise, specifies how the weights will be updated.
	\param [in] param The parameters of the algorithm.
	
	When we misclassify a data point during training, this function is used to update the existing weight-vector. It brings the true-class weight closer to the misclassified 
	data point, and to push the winning other-class weight away from the misclassified point according to AMM weight-update equations. The missclassified example used to update
	an existing weight is located in the input data set loaded to budgetedData.
*/
void budgetedVectorAMM::updateUsingDataPoint(budgetedData* inputData, unsigned int oto, unsigned int t, int sign, parameters *param)
{
	unsigned long pointIndexPointer = inputData->ai[t];
	unsigned long maxPointIndex = ((t + 1) == inputData->ai.size()) ? inputData->aj.size() : inputData->ai[t + 1];
	
	long double linKern = this->linearKernel(t, inputData, param);
	long double divisor = (long double)sign * ((long double)oto + 1.0) * (long double)(*param).LAMBDA_PARAM * degradation;
	for (unsigned long i = pointIndexPointer; i < maxPointIndex; i++)
	{
		((*this)[inputData->aj[i] - 1]) = (float)((long double)((*this)[inputData->aj[i] - 1]) + (long double)inputData->an[i] / divisor);
	}
    if ((*param).BIAS_TERM != 0)
	{
		((*this)[(*param).DIMENSION - 1]) = (float)((long double)((*this)[(*param).DIMENSION - 1]) + (long double)(*param).BIAS_TERM / divisor);
	}
	
	this->sqrL2norm += (long double)inputData->getVectorSqrL2Norm(t, param) / (divisor * divisor) + 2.0L / (divisor * this->degradation) * linKern;
}

/* \fn void budgetedVectorAMM::updateUsingVector(budgetedVectorAMM* otherVector, unsigned int oto, int sign, parameters *param)
	\brief Updates a weight-vector when misclassification happens.
	\param [in] otherVector Misclassified example used to update the existing weight.
	\param [in] oto Total number of iterations so far.
	\param [in] sign +1 if the input vector is of the true class, -1 otherwise, specifies how the weights will be updated.
	\param [in] param The parameters of the algorithm.
	
	When we misclassify a data point during training, this function is used to update the existing weight-vector. It brings the true-class weight closer to the misclassified 
	data point, and to push the winning other-class weight away from the misclassified point according to AMM weight-update equations. The missclassified example used to update
	an existing weight is located in the budgetedVectorAMM object.
*/
void budgetedVectorAMM::updateUsingVector(budgetedVectorAMM* otherVector, unsigned int oto, int sign, parameters *param)
{
	unsigned long chunkSize = chunkWeight;
	unsigned int i, j;
	float *tempArray = NULL;
	long double divisor = (long double)sign * ((long double)oto + 1.0) * (long double)(*param).LAMBDA_PARAM * degradation;
	long double linKern = this->linearKernel(otherVector);
	for (i = 0; i < arrayLength; i++)
	{
		// if the input vector's i-th array is NULL, then there is no need to update any of this vector's features
		if (otherVector->array[i] == NULL)
			continue;
		
		// now we know that i-th vector chunk of input vector has non-zero elements, go one by one and this vector
		if ((i == (arrayLength - 1)) && (dimension != chunkWeight))
			chunkSize = dimension % chunkWeight;
		
		// if the i-th chunk weight is NULL then create it
		if (this->array[i] == NULL)
		{
			// create and null the array
			tempArray = new (nothrow) float[chunkSize];
			for (j = 0; j < chunkSize; j++)
				*(tempArray + j) = 0;
			this->array[i] = tempArray;
		}
		else
			tempArray = this->array[i];
		
		for (j = 0; j < chunkSize; j++)
		{
			*(tempArray + j) += (float)((long double) otherVector->array[i][j] / divisor);
		}
		tempArray = NULL;
	}
	
	sqrL2norm += (long double)otherVector->getSqrL2norm() / (divisor * divisor) + 2.0L / (divisor * this->degradation) * linKern;
}

/* \fn float predictAMM(budgetedData *testData, parameters *param, budgetedModelAMM *model, vector <char> *labels)
	\brief Given a multi-hyperplane machine (MM) model, predict the labels of testing data.
	\param [in] testData Input test data.
	\param [in] param The parameters of the algorithm.
	\param [in] model Trained MM model.
	\param [out] labels Vector of predicted labels.
	\return Testing set error rate.
	
	Given the learned multi-hyperplane machine, the function computes the predictions on the testing data, outputing the predicted labels and the error rate.
*/
float predictAMM(budgetedData *testData, parameters *param, budgetedModelAMM *model, vector <char> *labels)
{
    unsigned long N, err = 0, totalPoints = 0;
	long double fx, maxFx;
	bool stillChunksLeft = true;
	long start, timeCalc = 0;
	char text[1024];
	budgetedVectorAMM *currentData = NULL;
	
	while (stillChunksLeft)
	{ 
        stillChunksLeft = testData->readChunk((*param).CHUNK_SIZE);
		(*param).updateVerySparseDataParameter(testData->getSparsity());
		
        N = testData->N;
		start = clock();    	
		for (unsigned int r = 0; r < N; r++)
    	{
			totalPoints++;
    		int y = 0;
    		maxFx = -INF;
			
			if ((*param).VERY_SPARSE_DATA)
			{
				// compute kernels using vectors directly from the budgetedData				
				for (unsigned int i = 0; i < (testData->yLabels).size(); i++)
				{
					for (unsigned int j = 0; j < (*(model->getModel()))[i].size(); j++)
					{
						fx = (*(model->getModel()))[i][j]->linearKernel(r, testData, param);
						if (fx > maxFx)
						{
							maxFx = fx;
							y = i;
						}
					}
				}	
			}
			else
			{
				// first create the budgetedVector using the vector from budgetedData, to be used in gaussianKernel() method below
				currentData = new budgetedVectorAMM((*param).DIMENSION, (*param).CHUNK_WEIGHT);
				currentData->budgetedVector::createVectorUsingDataPoint(testData, r, param);
				
				for (unsigned int i = 0; i < (testData->yLabels).size(); i++)
				{
					for (unsigned int j = 0; j < (*(model->getModel()))[i].size(); j++)
					{
						fx = (*(model->getModel()))[i][j]->linearKernel(currentData);
						if (fx > maxFx)
						{
							maxFx = fx;
							y = i;
						}
					}
				}
				delete currentData;
				currentData = NULL;
			}
			
			// save predicted label, will be sent to output
			if (labels)
				(*labels).push_back((char)(testData->yLabels)[y]);
			
    		if (y != testData->al[r])
    			err++;
    	}
		
		timeCalc += clock() - start;
    	
		if (((*param).VERBOSE) && (N > 0))
    	{
			sprintf(text, "Number of examples processed: %ld\n", totalPoints);
			svmPrintString(text);
        }
    }
	
	testData->flushData();
	
	if ((*param).VERBOSE)
    {		
		sprintf(text, "*** Testing completed in %5.3f seconds\n*** Testing error rate: %3.2f percent\n\n", (double)timeCalc / (double)CLOCKS_PER_SEC, 100.0 * (double)err / (double)totalPoints);
		svmPrintString(text);
    }
	
	return (float) (100.0 * (float)err / (float)totalPoints);
}

/* \fn void trainPegasos(budgetedData *trainData, parameters *param, budgetedModelAMM *model)
	\brief Train Pegasos.
	\param [in] trainData Input training data.
	\param [in] param The parameters of the algorithm.
	\param [in,out] model Initial Pegasos model.
	
	The function trains Pegasos model, given input data, the initial model (most often zero-weight model), and the parameters of the model.
*/
void trainPegasos(budgetedData *trainData, parameters *param, budgetedModelAMM *model)
{
	unsigned int sizeOfyLabels = 0, numIter = 0, t, i1, i2 = 0, N;
	unsigned long timeCalc = 0, start;
	long double fx, fx1, fx2, maxFx;
	bool stillChunksLeft = true;
	char text[1024];
	budgetedVectorAMM *currentData = NULL;
	
	// train the model
	for (unsigned int epoch = 0; epoch < (*param).NUM_EPOCHS; epoch++)
	{
		stillChunksLeft = true;
		while (stillChunksLeft)
		{ 
            stillChunksLeft = trainData->readChunk((*param).CHUNK_SIZE);
			(*param).updateVerySparseDataParameter(trainData->getSparsity());
			
            N = trainData->N;			
			if (numIter == 0)
			{				
				// initialize the model
				sizeOfyLabels = (unsigned int) trainData->yLabels.size();
				for (unsigned int i = 0; i < sizeOfyLabels; i++)
				{
					currentData = new budgetedVectorAMM((*param).DIMENSION, (*param).CHUNK_WEIGHT);
					vector <budgetedVectorAMM*> perClassWeights;
					perClassWeights.push_back(currentData);
					currentData = NULL;
					(*((*model).getModel())).push_back(perClassWeights);
				}
			}
			else if (sizeOfyLabels != (unsigned int) trainData->yLabels.size())
			{
				// if in the chunks before some class wasn't observed add it here; could happen with small chunks or unbalanced classes
				for (unsigned int i = 0; i < (trainData->yLabels.size() - sizeOfyLabels); i++)
				{
					currentData = new budgetedVectorAMM((*param).DIMENSION, (*param).CHUNK_WEIGHT);
					vector <budgetedVectorAMM*> perClassWeights;
					perClassWeights.push_back(currentData);
					currentData = NULL;
					(*((*model).getModel())).push_back(perClassWeights);
				}				
				sizeOfyLabels = (unsigned int) trainData->yLabels.size();
			}
            
            vector <unsigned int> tv(N, 0);
        	for (unsigned int ti = 0; ti < N; ti++)
        	{
        		tv[ti] = ti;
        	}
        	
        	// randomize the data
       	    random_shuffle(tv.begin(), tv.end());
        	
            start = clock();
            for (unsigned int ot = 0; ot < N; ot++)
            {
				numIter++;
                t = tv[ot];
				
    			i1 = trainData->al[t];
				if ((*param).VERY_SPARSE_DATA)
				{
					// compute kernels using vectors directly from the budgetedData
					fx1 = (*((*model).getModel()))[i1][0]->linearKernel(t, trainData, param);
				}
				else
				{
					// first create the budgetedVector using the vector from budgetedData, to be used in linearKernel() method below
					currentData = new budgetedVectorAMM((*param).DIMENSION, (*param).CHUNK_WEIGHT);
					currentData->budgetedVector::createVectorUsingDataPoint(trainData, t, param);
					
					fx1 = (*((*model).getModel()))[i1][0]->linearKernel(currentData);
				}
    			    
    			//calculate i-, fi-
    			fx = 0;
    			maxFx = -INF;
    			for (unsigned int i = 0; i < sizeOfyLabels; i++)
    			{
    				if (i == i1)
    					continue;
					
					if ((*param).VERY_SPARSE_DATA)
						fx = (*((*model).getModel()))[i][0]->linearKernel(t, trainData, param);
					else
						fx = (*((*model).getModel()))[i][0]->linearKernel(currentData);
    				
					if (fx > maxFx)
    				{
    					maxFx = fx;
    					i2 = i;
    				}
    			}
    			fx2 = maxFx;
    
    			// downgrade the weights
				for (unsigned int i = 0; i < sizeOfyLabels; i++)
				{
					(*((*model).getModel()))[i][0]->downgrade(numIter);
				}
    
    			// calculate the margin, if misclassified update weights
    			if (1.0L + fx2 - fx1 > 0.0L)
    			{
					if ((*param).VERY_SPARSE_DATA)
					{
						(*((*model).getModel()))[i2][0]->updateUsingDataPoint(trainData, numIter, t, -1, param);
						(*((*model).getModel()))[i1][0]->updateUsingDataPoint(trainData, numIter, t, 1, param);
					}
					else
					{
						(*((*model).getModel()))[i2][0]->updateUsingVector(currentData, numIter, -1, param);
						(*((*model).getModel()))[i1][0]->updateUsingVector(currentData, numIter, 1, param);
					}
    			}
				
				if (!(*param).VERY_SPARSE_DATA)
				{
					// if sparse data then no need for this part, since we didn't even create currentData
					delete currentData;
					currentData = NULL;
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
	trainData->flushData();
	
	if ((*param).VERBOSE)
    {
		sprintf(text, "*** Training completed in %5.3f seconds.\n", (double)timeCalc / (double)CLOCKS_PER_SEC);
		svmPrintString(text);
    }
}

/* \fn void trainAMMonline(budgetedData *trainData, parameters *param, budgetedModelAMM *model)
	\brief Train AMM online.
	\param [in] trainData Input training data.
	\param [in] param The parameters of the algorithm.
	\param [in,out] model Initial AMM model.
	
	The function trains multi-hyperplane machine using AMM online algotihm, given input data, the initial model (most often zero-weight model), and the parameters of the model.
*/
void trainAMMonline(budgetedData *trainData, parameters *param, budgetedModelAMM *model)
{	
	vector <unsigned int> n;
	unsigned long timeCalc = 0, start; 
	long double fx1, fx2, maxFx;
	unsigned int sizeOfyLabels = 0, countNew = 0, countDel = 0, numIter = 0, i1, i2, j1, j2, t, N;
	bool stillChunksLeft = true;
	char text[1024];
	budgetedVectorAMM *currentData = NULL;
	
	// train the model
	for (unsigned int epoch = 0; epoch < (*param).NUM_EPOCHS; epoch++)
	{
		stillChunksLeft = true;
		while (stillChunksLeft)
		{
            stillChunksLeft = trainData->readChunk((*param).CHUNK_SIZE);
			(*param).updateVerySparseDataParameter(trainData->getSparsity());
			
            N = trainData->N;			
			if (numIter == 0)
			{
				// initialize the model with zero weights
				sizeOfyLabels = (unsigned int) trainData->yLabels.size();
				for (unsigned int i = 0; i < sizeOfyLabels; i++)
				{
					n.push_back(1);

					currentData = new budgetedVectorAMM((*param).DIMENSION, (*param).CHUNK_WEIGHT);
					vector <budgetedVectorAMM*> v1;
					v1.push_back(currentData);
					currentData = NULL;
					(*((*model).getModel())).push_back(v1);
				}
			}
			else if (sizeOfyLabels != trainData->yLabels.size())
			{
				// if in the chunks before some class wasn't observed, could happen with small chunks or unbalanced classes
				for (unsigned int i = 0; i < (trainData->yLabels.size() - sizeOfyLabels); i++)
				{
					currentData = new budgetedVectorAMM((*param).DIMENSION, (*param).CHUNK_WEIGHT);
					vector <budgetedVectorAMM*> perClassWeights;
					perClassWeights.push_back(currentData);
					currentData = NULL;
					(*((*model).getModel())).push_back(perClassWeights);
				}				
				sizeOfyLabels = (unsigned int) trainData->yLabels.size();
			}
            
            // randomize
            vector <unsigned int> tv(N, 0);
        	for (unsigned int ti = 0; ti < N; ti++)
        	{
        		tv[ti] = ti;
        	}
        	random_shuffle(tv.begin(), tv.end());
        	
            start = clock();
            for (unsigned int ot = 0; ot < N; ot++)
            {    
				numIter++;
    			t = tv[ot];
				
				if (!(*param).VERY_SPARSE_DATA)
				{
					// only create currentData if the data is non-sparse, otherwise kernels will be computed directly from trainData
					currentData = new budgetedVectorAMM((*param).DIMENSION, (*param).CHUNK_WEIGHT);
					currentData->budgetedVector::createVectorUsingDataPoint(trainData, t, param);
				}
				
                //calculate i+,j+
    			i1 = trainData->al[t];
    			j1 = 0;
    			maxFx = -INF;
    			
    			for (unsigned int j = 0; j < n[i1]; j++)
    			{
					if ((*param).VERY_SPARSE_DATA)
						fx1 = (*((*model).getModel()))[i1][j]->linearKernel(t, trainData, param);
					else
						fx1 = (*((*model).getModel()))[i1][j]->linearKernel(currentData);
					
    				if (fx1 > maxFx)
    				{
    					j1 = j;
    					maxFx = fx1;
    				}
    			}
    			fx1 = maxFx;
				
    			// calculate i-, j-
    			i2 = 0;
    			j2 = 0;
    			fx2 = 0;
    			maxFx = -INF;
    			for (unsigned int i = 0; i < sizeOfyLabels; i++)
    			{
    				if (i == i1)
    					continue;
    					
    				for (unsigned int j = 0; j < n[i]; j++)
    				{
						if ((*param).VERY_SPARSE_DATA)
							fx2 = (*((*model).getModel()))[i][j]->linearKernel(t, trainData, param);
						else
							fx2 = (*((*model).getModel()))[i][j]->linearKernel(currentData);
						
    					if (fx2 > maxFx)
    					{
    						maxFx = fx2;
    						i2 = i;
    						j2 = j;
    					}
    				}
    			}    	        
    			fx2 = maxFx;
    
    			// downgrade weights each iteration
				for (unsigned int i = 0; i < sizeOfyLabels; i++)
					for (unsigned int j = 0; j < n[i]; j++)
						(*((*model).getModel()))[i][j]->downgrade(numIter);
				
    			if (1.0 + fx2 - fx1 > 0.0)
    			{
					// we made a misprediction, push negative class further away, and positive closer!					
					if ((*param).VERY_SPARSE_DATA)
					{
						// since we did not create currentData earlier, here we create it to perform updates
						currentData = new budgetedVectorAMM((*param).DIMENSION, (*param).CHUNK_WEIGHT);
						currentData->budgetedVector::createVectorUsingDataPoint(trainData, t, param);
					}
					
					// push the other class further away
					(*((*model).getModel()))[i2][j2]->updateUsingVector(currentData, numIter, -1, param);
					
					// update the true class weight
					if (fx1 > 0.0)
    				{
						(*((*model).getModel()))[i1][j1]->updateUsingVector(currentData, numIter, 1, param);
						
						delete currentData;
						currentData = NULL;
    				} 
                    else 
                    {                         
						if (n[i1] < (*param).LIMIT_NUM_WEIGHTS_PER_CLASS) // limit number of weights (we found ~20 is a reasonable number per class)
						{
        					n[i1]++;                            
        					currentData->updateDegradation(numIter, param);                            
        					(*((*model).getModel()))[i1].push_back(currentData);                            
        					countNew++;							
							currentData = NULL;
                        }
						else
						{
							delete currentData;
							currentData = NULL;
						}
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
    	        
    			// pruning phase
    			if (numIter % (int)(*param).K_PARAM == 0)
    			{
                    long double sumNorms = 0, sumThreshold = (long double)(*param).C_PARAM * (long double)(*param).C_PARAM / ((long double)numIter * (long double)numIter * (*param).LAMBDA_PARAM * (*param).LAMBDA_PARAM);              
                    vector <long double> weightNorms, sortedWeightNorms;
                    int numToDelete = 0; 
                    
                    // first find the norms of weights   
        			for (unsigned int i = 0; i < sizeOfyLabels; i++)
					{
        				for (vector<budgetedVectorAMM*>::iterator vi = (*((*model).getModel()))[i].begin(); vi != (*((*model).getModel()))[i].end(); vi++)
						{
							weightNorms.push_back((double) (*(*vi)).getSqrL2norm());
						}
					}
					
        			// now sort them
        			sortedWeightNorms = weightNorms;
        			sort(sortedWeightNorms.begin(), sortedWeightNorms.end()); 
        			    
        			// find how many before threshold exceeded
                    for (unsigned int i = 0; i < weightNorms.size(); i++)
                    {
                        sumNorms += sortedWeightNorms[i];
                        if (sumNorms > sumThreshold)
                           break;
                        else
                           numToDelete++;                             
                    }
                    
                    // delete those that should be deleted
                    int counter = 0;
                    bool deleted = false;
                    for (unsigned int i = 0; i < sizeOfyLabels; i++)
    				{
    					for (vector<budgetedVectorAMM*>::iterator vi = (*((*model).getModel()))[i].begin(); vi != (*((*model).getModel()))[i].end();)
                        {
                            long double currNorm = weightNorms[counter++];
                            
                            deleted = false;
                            for (int j = 0; j < numToDelete; j++)
                            {
                                if (currNorm == sortedWeightNorms[j])
                                {
									if (n[i] == 1)
									{
										svmPrintString("Was about to delete all weights of a class, check K_PARAM and C_PARAM parameters!\n");
										break;
									}
									
                                    delete (*vi);
                                    vi = (*((*model).getModel()))[i].erase(vi);
        							n[i]--;
									
                                    countDel++;
        							deleted = true;
        							break;
                                }
                            }
                            
                            if (!deleted)
								vi++;
    					}
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
	trainData->flushData();

    if ((*param).VERBOSE)
    {
		sprintf(text, "*** Training completed in %5.3f seconds.\nNumber of weights deleted: %d\n", (double)timeCalc / (double)CLOCKS_PER_SEC, countDel);
		svmPrintString(text);
		for (unsigned int i = 0; i < sizeOfyLabels; i++)
		{
			sprintf(text, "Number of weights of class %d: %d\n", i + 1, n[i]);
			svmPrintString(text);
		}
    }
}

/* \fn void trainAMMbatch(budgetedData *trainData, parameters *param, budgetedModelAMM *model)
	\brief Train AMM batch.
	\param [in] trainData Input training data.
	\param [in] param The parameters of the algorithm.
	\param [in,out] model Initial AMM model.
	
	The function trains multi-hyperplane machine using AMM batch algotihm, given input data, the initial model (most often zero-weight model), and the parameters of the model.
*/
void trainAMMbatch(budgetedData *trainData, parameters *param, budgetedModelAMM *model)
{
	vector <unsigned int> n;	// stores number of weights per class
	unsigned long timeCalc = 0, start;
	long double fx1, fx2, maxFx, assocFx;
	unsigned int i, j, t, N, i1, i2, j1, j2, sizeOfyLabels = 0, countNew = 0, countDel = 0, numIter = 0, currAssign = 0, currAssignID;
	bool stillChunksLeft;
	char text[1024];
    budgetedVectorAMM *currentData = NULL;
	
    //Initialization phase with algorithm AMM_online
	stillChunksLeft = true;
	while (stillChunksLeft)
	{ 
        stillChunksLeft = trainData->readChunk((*param).CHUNK_SIZE);
		(*param).updateVerySparseDataParameter(trainData->getSparsity());
		
        N = trainData->N;		
		if (numIter == 0)
		{
			//Initialize
			sizeOfyLabels = (unsigned int) trainData->yLabels.size();
			for (i = 0; i < sizeOfyLabels; i++)
			{
				n.push_back(1);

				currentData = new budgetedVectorAMM((*param).DIMENSION, (*param).CHUNK_SIZE);
				vector <budgetedVectorAMM*> perClassWeights;
				perClassWeights.push_back(currentData);
				currentData = NULL;
				(*((*model).getModel())).push_back(perClassWeights);
			}
		}
		else if (sizeOfyLabels != (unsigned int) trainData->yLabels.size())
		{
			// if in previous chunks some class wasn't observed, could happen with small chunks or unbalanced classes
			// just add new zero weights for the new classes
			for (i = 0; i < (trainData->yLabels.size() - sizeOfyLabels); i++)
			{
				currentData = new budgetedVectorAMM((*param).DIMENSION, (*param).CHUNK_WEIGHT);
				vector <budgetedVectorAMM*> perClassWeights;
				perClassWeights.push_back(currentData);
				currentData = NULL;
				(*((*model).getModel())).push_back(perClassWeights);
			}				
			sizeOfyLabels = (unsigned int) trainData->yLabels.size();
		}
        
        // randomize
        vector <unsigned int> tv(N, 0);
        unsigned int *assigns = new unsigned int[N];
    	for (i = 0; i < N; i++)
    	{
    		tv[i] = i;
    	}
    	random_shuffle(tv.begin(), tv.end());
        
        start = clock();
    	for (unsigned int trainIter = 0; trainIter < N; trainIter++)
    	{
			numIter++;
    		t = tv[trainIter];
			
			if (!(*param).VERY_SPARSE_DATA)
			{
				// only create currentData if the data is non-sparse, otherwise kernels will be computed directly from trainData
				currentData = new budgetedVectorAMM((*param).DIMENSION, (*param).CHUNK_WEIGHT);
				currentData->budgetedVector::createVectorUsingDataPoint(trainData, t, param);
			}
			
    		// calculate i+, j+
    		i1 = trainData->al[t];
    		j1 = 0;
    		maxFx = -INF;
    		for (j = 0; j < n[i1]; j++)
    		{
				if ((*param).VERY_SPARSE_DATA)
					fx1 = (*((*model).getModel()))[i1][j]->linearKernel(t, trainData, param);
				else
					fx1 = (*((*model).getModel()))[i1][j]->linearKernel(currentData);
				
    			if (fx1 > maxFx)
    			{
    				j1 = j;
    				maxFx = fx1;
    			}
    		}
    		fx1 = maxFx;
    		*(assigns + t) = (*((*model).getModel()))[i1][j1]->getID();
    
    		// calculate i-, j-
    		i2 = 0;
    		j2 = 0;
    		fx2 = 0;
    		maxFx = -INF;
    		for (i = 0; i < sizeOfyLabels; i++)
    		{
    			if (i == i1)
    				continue;
    				
    			for (j = 0; j < n[i]; j++)
    			{
					if ((*param).VERY_SPARSE_DATA)
						fx2 = (*((*model).getModel()))[i][j]->linearKernel(t, trainData, param);
					else
						fx2 = (*((*model).getModel()))[i][j]->linearKernel(currentData);
					
    				if (fx2 > maxFx)
    				{
    					maxFx = fx2;
    					i2 = i;
    					j2 = j;
    				}
    			}
    		}
    		fx2 = maxFx;
    
    		// downgrade weight each iteration
			for (i = 0; i < sizeOfyLabels; i++)
				for (j = 0; j < n[i]; j++)
					(*((*model).getModel()))[i][j]->downgrade(numIter);
			
    		if (1.0 + fx2 - fx1 > 0.0)
    		{
				// we made a misprediction, push negative class further away, and positive closer!
				if ((*param).VERY_SPARSE_DATA)
				{
					// since we did not create currentData earlier, here we create it to perform updates
					currentData = new budgetedVectorAMM((*param).DIMENSION, (*param).CHUNK_WEIGHT);
					currentData->budgetedVector::createVectorUsingDataPoint(trainData, t, param);
				}
				
				// push the other class further away
    			(*((*model).getModel()))[i2][j2]->updateUsingVector(currentData, numIter, -1, param);
				
				// update the true class weight
    			if (fx1 > 0.0)
    			{
    				(*((*model).getModel()))[i1][j1]->updateUsingVector(currentData, numIter, 1, param);
					
					delete currentData;
					currentData = NULL;
    			} 
                else 
                {
    				if (n[i1] < (*param).LIMIT_NUM_WEIGHTS_PER_CLASS)
                    {
                        n[i1]++;
        				currentData->updateDegradation(numIter, param);
        				(*((*model).getModel()))[i1].push_back(currentData);
        				currentData = NULL;
        				countNew++;
                    }
					else
					{
						delete currentData;
						currentData = NULL;
					}
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
		
    	trainData->saveAssignment(assigns);
    	delete [] assigns;
        
		if (((*param).VERBOSE) && (N > 0))
		{
			sprintf(text, "Number of examples processed: %d\n", numIter);
			svmPrintString(text);
        }    	
    }	
    
	if ((*param).VERBOSE)
		svmPrintString("Initialization epoch done!\n");
    
	// end of init phase, start AMM algorithm below
	
	for (unsigned int epoch = 1; epoch <= (*param).NUM_EPOCHS; epoch++)
	{
		stillChunksLeft = true;
		while (stillChunksLeft)
		{
            stillChunksLeft = trainData->readChunk((*param).CHUNK_SIZE, true);
			(*param).updateVerySparseDataParameter(trainData->getSparsity());
			
            N = trainData->N;            
            trainData->readChunkAssignments(!stillChunksLeft);
            
            // randomize
            vector <int> tv(N, 0);
        	for (unsigned int ti = 0; ti < N; ti++)
        		tv[ti] = ti;
        		
        	random_shuffle(tv.begin(), tv.end());        	
            start = clock();            
    		for (unsigned int trainIter = 0; trainIter < N; trainIter++)
    		{
				numIter++;
    			t = tv[trainIter];
    			
				if (!(*param).VERY_SPARSE_DATA)
				{
					currentData = new budgetedVectorAMM((*param).DIMENSION, (*param).CHUNK_WEIGHT);
					currentData->budgetedVector::createVectorUsingDataPoint(trainData, t, param);
				}
				
    			// calculate i+, j+
    			i1 = trainData->al[t];
    			j1 = 0;
    			
    			currAssignID = trainData->assignments[t];
    			
    			maxFx = 0;
    			assocFx = -INF;
    			for (j = 0; j < n[i1]; j++)
    			{
					if ((*param).VERY_SPARSE_DATA)
						fx1 = (*((*model).getModel()))[i1][j]->linearKernel(t, trainData, param);
					else
						fx1 = (*((*model).getModel()))[i1][j]->linearKernel(currentData);
					
    				if ((maxFx == 0) || (fx1 > maxFx))
    				{
    					j1 = j;
    					maxFx = fx1;
    				}
    				
    				// this is the prediction of the associated same-label weight
    				if ((*((*model).getModel()))[i1][j]->getID() == currAssignID)
    				{
                       currAssign = j;
                       assocFx = fx1;
                    }
    			}
    			
    			fx1 = maxFx;
    			if (assocFx == -INF)
 			    {
                    assocFx = maxFx;
                    currAssign = j1;                             
                }
    
    			// calculate i-, j-
    			i2 = 0;
    			j2 = 0;
    			fx2 = 0;
    			maxFx = 0;
    			for (i = 0; i < sizeOfyLabels; i++)
    			{
    				if (i == i1)
    					continue;
    					
    				for (j = 0; j < n[i]; j++)
    				{
						if ((*param).VERY_SPARSE_DATA)
							fx2 = (*((*model).getModel()))[i][j]->linearKernel(t, trainData, param);
						else
							fx2 = (*((*model).getModel()))[i][j]->linearKernel(currentData);
						
    					if ((maxFx == 0) || (fx2 > maxFx))
    					{
    						maxFx = fx2;
    						i2 = i;
    						j2 = j;
    					}
    				}
    			}    			
    			fx2 = maxFx;
    			
    			// downgrade weights each iteration
				for (unsigned int i = 0; i < sizeOfyLabels; i++)
				{
					for (unsigned int j = 0; j < n[i]; j++)
					{
						(*((*model).getModel()))[i][j]->downgrade(numIter);
					}
				}
				
    			//calculate v
    			if (1.0 + fx2 - assocFx > 0.0)
    			{
					// we made a misprediction, update the weights by pushing the wrong-class weight further from the misclassified
					//	example, and the true-class closer to the misclassified example
					if ((*param).VERY_SPARSE_DATA)
					{
						// since we did not create currentData earlier, here we create it to perform updates
						currentData = new budgetedVectorAMM((*param).DIMENSION, (*param).CHUNK_WEIGHT);
						currentData->budgetedVector::createVectorUsingDataPoint(trainData, t, param);
					}
					
					(*((*model).getModel()))[i1][currAssign]->updateUsingVector(currentData, numIter, 1, param);
					(*((*model).getModel()))[i2][j2]->updateUsingVector(currentData, numIter, -1, param);
    				if ((fx1 <= 0.0) && (n[i1] < (*param).LIMIT_NUM_WEIGHTS_PER_CLASS))
                    {
						n[i1]++;
						currentData->updateDegradation(numIter, param);
						(*((*model).getModel()))[i1].push_back(currentData);
						currentData = NULL;
						countNew++;
    				}
					else
					{
						// if over the budget, we do not add a new data point to the budget
						delete currentData;
						currentData = NULL;
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
    
    			timeCalc += clock() - start;
    			start = clock();
    			
    			if (numIter % (*param).K_PARAM == 0)
    			{
                    // we run the pruning procedure here
                    long double sumNorms = 0; 
					long double sumThreshold = (long double)(*param).C_PARAM * (long double)(*param).C_PARAM / ((long double)numIter * (long double)numIter * (*param).LAMBDA_PARAM * (*param).LAMBDA_PARAM);        
                    vector <long double> weightNorms, sortedWeightNorms;
                    int numToDelete = 0;  
                    
                    // first find the norms of weights
        			for (unsigned int i = 0; i < sizeOfyLabels; i++)
					{
        				for (vector<budgetedVectorAMM*>::iterator vi = (*((*model).getModel()))[i].begin(); vi != (*((*model).getModel()))[i].end(); vi++)
						{
							weightNorms.push_back((double) (*(*vi)).getSqrL2norm());
						}
					}
					
        			// now sort them
        			sortedWeightNorms = weightNorms;
        			sort(sortedWeightNorms.begin(), sortedWeightNorms.end());
        			    
        			// find how many before threshold is exceeded
                    for (unsigned int i = 0; i < weightNorms.size(); i++)
                    {
                        sumNorms += sortedWeightNorms[i];
                        if (sumNorms > sumThreshold)
                           break;
                        else
                           numToDelete++;                             
                    }
                    
                    // delete those that should be deleted, with aggregate norm less than the set threshold
                    int counter = 0;
                    bool deleted = false;
                    for (unsigned int i = 0; i < sizeOfyLabels; i++)
        			{
        				for (vector<budgetedVectorAMM*>::iterator vi = (*((*model).getModel()))[i].begin(); vi != (*((*model).getModel()))[i].end();)
                        {
                            long double currNorm = weightNorms[counter++];
                            
                            deleted = false;
                            for (int j = 0; j < numToDelete; j++)
                            {
                                if (currNorm == sortedWeightNorms[j])
                                {
									if (n[i] == 1)
									{
										svmPrintString("Was about to delete all weights of a class, check the K_PARAM and C_PARAM parameters!\n");
										break;
									}
									
                                    delete (*vi);
                                    vi = (*((*model).getModel()))[i].erase(vi);
        							n[i]--;
									
        							countDel++;
        							deleted = true;
        							break;
                                }
                            }
                            
                            if (!deleted)
                               vi++;
        				}
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
        
        // every so-so (around 3) subepochs recalculate the associations
		if (((epoch % (*param).NUM_SUBEPOCHS) == 0) && (epoch != (*param).NUM_EPOCHS))
		{		
            // calculate the new assignments
            stillChunksLeft = true;
    		while (stillChunksLeft)
    		{
                stillChunksLeft = trainData->readChunk((*param).CHUNK_SIZE);
                N = trainData->N;
                unsigned int *assigns = new unsigned int[N];
            	
                start = clock();
                for (unsigned int ot = 0; ot < N; ot++)
            	{
            		t = ot;
					if ((*param).VERY_SPARSE_DATA)
					{
						// calculate i+, j+
						i1 = trainData->al[t];
						j1 = 0;
						maxFx = -INF;
						for (unsigned int j = 0; j < n[i1]; j++)
						{
							fx1 = (*((*model).getModel()))[i1][j]->linearKernel(t, trainData, param);
							if (fx1 > maxFx)
							{
								j1 = j;
								maxFx = fx1;
							}
						}
					}
					else
					{
						currentData = new budgetedVectorAMM((*param).DIMENSION, (*param).CHUNK_WEIGHT);
						currentData->budgetedVector::createVectorUsingDataPoint(trainData, t, param);
						
						// calculate i+, j+
						i1 = trainData->al[t];
						j1 = 0;
						maxFx = -INF;
						for (unsigned int j = 0; j < n[i1]; j++)
						{
							fx1 = (*((*model).getModel()))[i1][j]->linearKernel(currentData);
							if (fx1 > maxFx)
							{
								j1 = j;
								maxFx = fx1;
							}
						}
						delete currentData;
						currentData = NULL;
					}
					
            		*(assigns + t) = (*((*model).getModel()))[i1][j1]->getID();
                }
                timeCalc += clock() - start;
				
                trainData->saveAssignment(assigns);
	            delete [] assigns;                
			}		
		}
		
		if ((*param).VERBOSE && ((*param).NUM_EPOCHS > 1))
		{
			sprintf(text, "Epoch %d/%d done.\n", epoch, (*param).NUM_EPOCHS);
			svmPrintString(text);
		}
	}
    trainData->flushData();
    
    if ((*param).VERBOSE)
    {
        sprintf(text, "*** Training completed in %5.3f seconds.\nNumber of weights deleted: %d\n", (double) timeCalc / (double) CLOCKS_PER_SEC, countDel);
		svmPrintString(text);
		for (unsigned int i = 0; i < sizeOfyLabels; i++)
		{
			sprintf(text, "Number of weights of class %d: %d\n", i + 1, n[i]);
			svmPrintString(text);
		}
    }
}
/*
	\file budgetedsvm-predict.cpp
	\brief Source file implementing commmand-prompt interface for testing phase of budgetedSVM library.
*/
/*
	* This program is a free software; you can redistribute it and/or modify
	* it under the terms of the GNU General Public License as published by
	* the Free Software Foundation; either version 3 of the License, or
	* (at your option) any later version.
	
	Author	:	Nemanja Djuric
	Name	:	budgetedsvm_train.cpp
	Date	:	November 19th, 2012
	Desc.	:	Source file implementing commmand-prompt interface for testing phase of budgetedSVM library.
*/

#include "../Eigen/Dense"
using namespace Eigen;

#include <vector>
#include <sstream>
#include <time.h>
#include <algorithm>
#include <stdio.h>
using namespace std;

#include "budgetedSVM.h"
#include "mm_algs.h"
#include "bsgd.h"
#include "llsvm.h"

int main(int argc, char **argv)
{	
	parameters param;
	if (argc == 1)
	{
		printUsagePrompt(false, &param);
		return 0;
	}
	
	// vars
	char inputFileName[1024];
	char modelFileName[1024];
	char outputFileName[1024];
	vector <int> yLabels;	
	vector <char> predLabels;
	budgetedModel *model = NULL;
	FILE *pFile = NULL;
	
	// parse input string
	parseInputPrompt(argc, argv, false, inputFileName, modelFileName, outputFileName, &param);
	param.ALGORITHM = budgetedModel::getAlgorithm(modelFileName);
	
	// init random number generator
	srand((unsigned)time(NULL));
	
	// initialize test data and run trained model
	budgetedData *testData = NULL;
	switch (param.ALGORITHM)
	{
		case PEGASOS:
		case AMM_BATCH:
		case AMM_ONLINE:
			// read the text file and populate model
			model = new budgetedModelAMM;
			if (!model->loadFromTextFile(modelFileName, &yLabels, &param))
			{
				printf("Error: can't read model from file %s.\n", modelFileName);
				delete model;
				return 1;
			}
			testData = new budgetedData(inputFileName, param.DIMENSION - (int) (param.BIAS_TERM != 0.0), param.CHUNK_SIZE, false, &yLabels);
			
			predictAMM(testData, &param, (budgetedModelAMM*) model, &predLabels);
			break;
			
		case LLSVM:
			model = new budgetedModelLLSVM;
			if (!model->loadFromTextFile(modelFileName, &yLabels, &param))
			{
				printf("Error: can't read model from file %s.\n", modelFileName);
				delete model;
				return 1;
			}
			testData = new budgetedData(inputFileName, param.DIMENSION - (int) (param.BIAS_TERM != 0.0), param.CHUNK_SIZE, false, &yLabels);
			
			predictLLSVM(testData, &param, (budgetedModelLLSVM*) model, &predLabels);
			break;
			
		case BSGD:
			// read and populate c++ model
			model = new budgetedModelBSGD;
			if (!model->loadFromTextFile(modelFileName, &yLabels, &param))
			{
				printf("Error: can't read model from file %s.\n", modelFileName);
				delete model;
				return 1;
			}
			testData = new budgetedData(inputFileName, param.DIMENSION - (int) (param.BIAS_TERM != 0.0), param.CHUNK_SIZE, false, &yLabels);
			
			predictBSGD(testData, &param, (budgetedModelBSGD*) model, &predLabels);
			break;
		default:
			printf("Error, algorithm not recognized.\n");
			return 1;
	}
	delete testData;
	delete model;
	
	// print labels to output file
	pFile = fopen(outputFileName, "wt");
	if (!pFile)
	{
		printf("Error writing to output file %s.\n", outputFileName);
		return 1;
	}
	
	for (unsigned int i = 0; i < predLabels.size(); i++)
		fprintf(pFile, "%d\n", predLabels[i]);
	fclose(pFile);
}
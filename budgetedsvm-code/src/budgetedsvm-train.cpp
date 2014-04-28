/*
	\file budgetedsvm-train.cpp
	\brief Source file implementing commmand-prompt interface for training phase of budgetedSVM library.
*/
/*
	* This program is a free software; you can redistribute it and/or modify
	* it under the terms of the GNU General Public License as published by
	* the Free Software Foundation; either version 3 of the License, or
	* (at your option) any later version.

	Author	:	Nemanja Djuric
	Name	:	budgetedsvm_train.cpp
	Date	:	November 30th, 2012
	Desc.	:	Source file implementing commmand-prompt interface for training phase of budgetedSVM library.
*/

#include "../Eigen/Dense"
using namespace Eigen;

#include <vector>
#include <sstream>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
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
		printUsagePrompt(true, &param);
		return 0;
	}

	// vars
	char inputFileName[1024];
	char modelFileName[1024];
	budgetedModel *model = NULL;
    int seed = 0;

	// parse input string
	parseInputPrompt(argc, argv, true, inputFileName, modelFileName, NULL, &param, &seed);

	// init random number generator
    if (seed == 0)
    {
        struct timeval time;
        gettimeofday(&time,NULL);
        // using this will allow launching multiple instances per second and not getting the same seed
        seed = (int) (time.tv_sec * 1000) + (int) (time.tv_usec / 1000) + getpid();
    }
    srand(seed);

	// train a model
	budgetedData *trainData = NULL;
	switch (param.ALGORITHM)
	{
		case PEGASOS:
			model = new budgetedModelAMM;
			trainData = new budgetedData(inputFileName, param.DIMENSION - (int) (param.BIAS_TERM != 0.0), param.CHUNK_SIZE);
			trainPegasos(trainData, &param, (budgetedModelAMM*) model);
			break;
		case AMM_BATCH:
			model = new budgetedModelAMM;
			trainData = new budgetedData(inputFileName, param.DIMENSION - (int) (param.BIAS_TERM != 0.0), param.CHUNK_SIZE, true);
			trainAMMbatch(trainData, &param, (budgetedModelAMM*) model);
			break;
		case AMM_ONLINE:
			model = new budgetedModelAMM;
			trainData = new budgetedData(inputFileName, param.DIMENSION - (int) (param.BIAS_TERM != 0.0), param.CHUNK_SIZE);
			trainAMMonline(trainData, &param, (budgetedModelAMM*) model);
			break;
		case LLSVM:
			model = new budgetedModelLLSVM;
			trainData = new budgetedData(inputFileName, param.DIMENSION - (int) (param.BIAS_TERM != 0.0), param.CHUNK_SIZE);
			trainLLSVM(trainData, &param, (budgetedModelLLSVM*) model);
			break;
		case BSGD:
			model = new budgetedModelBSGD;
			trainData = new budgetedData(inputFileName, param.DIMENSION - (int) (param.BIAS_TERM != 0.0), param.CHUNK_SIZE);
			trainBSGD(trainData, &param, (budgetedModelBSGD*) model);
			break;
	}

	// save model to .txt file
	model->saveToTextFile(modelFileName, &(trainData->yLabels), &param);
	delete model;

	// delete trainData, no more need for it
	delete trainData;
}

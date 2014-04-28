/*
	\file llsvm.cpp
	\brief Implementation of LLSVM algorithm.
	
	Implements the classes and functions used to train and test the LLSVM, low-rank linearization approach to large-scale SVM training.
*/
/*
	* This program is a free software; you can redistribute it and/or modify
	* it under the terms of the GNU General Public License as published by
	* the Free Software Foundation; either version 3 of the License, or
	* (at your option) any later version.
	
	Authors	:	Nemanja Djuric and Liang Lan
	Name	:	llsvm.cpp
	Date	:	November 20th, 2012
	Desc.	:	Implements the classes and functions used to train and test the LLSVM, low-rank linearization approach to large-scale SVM training.
	Version	:	v1.02, with added k-medoids, storing W matrix to outside model, and cleaner code
*/

#include "../Eigen/Dense"
using namespace Eigen;

#include <vector>
#include <time.h>
#include <sstream>
#include <iostream>
#include <math.h>
#include <algorithm>
#include <stdio.h>
using namespace std;

#include "budgetedSVM.h"
#include "llsvm.h"

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
template <class T> static inline void mySwap(T& x, T& y) { T t=x; x=y; y=t; }
template <class T> static inline T myMin(T x,T y) { return (x<y)?x:y; }
template <class T> static inline T myMax(T x,T y) { return (x>y)?x:y; }

/* prototypes of functions used in llsvm.cpp to find the mapping function; for details see the documentation for each function below
void invSquareRoot(MatrixXd &A);
MatrixXd sqDist(MatrixXd &A, MatrixXd &B);
void kMeans(MatrixXd &data, unsigned int numClusters, unsigned int maxIter, MatrixXd &center);
void kMedoids(budgetedData *trainData, parameters *param, unsigned int *medoidIndex);
void liblinear_Solve_l2r_l1(const MatrixXd &X, unsigned char *y, VectorXd &w, parameters *param, vector <int> *yLabels);
*/

/*! \fn void invSquareRoot(MatrixXd &A)
	\brief Computes square root of inverse of the input matrix.
	\param [in,out] A Matrix whose square root of inverse is to be computed.
*/
void invSquareRoot(MatrixXd &A)
{
	// the idea: A = U*V*U' --> A^(-0.5) = U*V.^(-0.5)
	SelfAdjointEigenSolver<MatrixXd> eigensolver(A);
	MatrixXd eigValW = eigensolver.eigenvalues().asDiagonal();
	MatrixXd eigVecW = eigensolver.eigenvectors();
	MatrixXd res = MatrixXd::Zero(eigValW.rows(), eigValW.cols());
	double maxVal = eigValW.maxCoeff();
	
	for (int i = 0; i < eigValW.rows(); i++)
		if (eigValW(i, i) > maxVal * 1E-5)
			res(i, i) = 1.0 / sqrt(eigValW(i, i));
	
	A = eigVecW * res;
}

/*! \fn MatrixXd sqDist(MatrixXd &A, MatrixXd &B)
	\brief Computes squared Euclidean distance between rows of matrices A and B.
	\param [in] A The first matrix with data points in rows.
	\param [in] B The second matrix with data points in rows.
	\return Matrix of distances between data points in A and data points in B.
	
	The function computes a matrix of distances between data points in A and data points in B. If the return matrix is C, then, e.g., C(0, 1) is a distance between the first row in matrix A and the second row in matrix B.
*/
MatrixXd sqDist(MatrixXd &A, MatrixXd &B)
{
	MatrixXd aa = A.rowwise().squaredNorm();
	MatrixXd bb = B.rowwise().squaredNorm();
	MatrixXd dist = (MatrixXd)(- 2.0 * A * B.transpose());
	
	for (int i = 0; i < dist.cols(); i++)
		dist.col(i) += aa;
	for (int i = 0; i < dist.rows(); i++)
		dist.row(i) += bb.transpose();
	
	dist = dist.cwiseAbs();
	return dist;
}
/*! \fn void kMeans(MatrixXd &data, unsigned int numClusters, unsigned int maxIter, MatrixXd &center)

	\brief Selects the landmark points from the input data using k-means algorithm.
	\param [in] data Data which is input to k-means.
	\param [in] numClusters Number of clusters that will be found.
	\param [in] maxIter Number of iterations of k-means.
	\param [out] center The cluster centers after k-means.
	
	The function selects the landmark points from the input data using k-means algorithm, used as an initialization of LLSVM.
*/
void kMeans(MatrixXd &data, unsigned int numClusters, unsigned int maxIter, MatrixXd &center)
{
	unsigned int N = (unsigned int) data.rows(), dim = (unsigned int) data.cols(), counter, iter, i, cluster;
	MatrixXd dist;
	
	// randomize
	vector <int> rnd(N, 0);
	for (i = 0; i < N; i++)
	{
		rnd[i] = i;
	}
	random_shuffle(rnd.begin(), rnd.end());
	
	// random select initial cluster centers
	for (i = 0; i < numClusters; i++)
	{
		center.row(i) = data.row(rnd[i]);
	}
	
	unsigned int *idx = new unsigned int[N];
	for (iter = 0; iter < maxIter; iter++)
	{
		dist = sqDist(center, data);
		
		MatrixXd::Index minRow, minCol;
		for (i = 0; i < N; i++)
		{
			dist.col(i).minCoeff(&minRow, &minCol);			
			idx[i] = (unsigned int) minRow;
		}
		
		for (cluster = 0; cluster < numClusters; cluster++)
		{	
			counter = 0;
			MatrixXd tempMean = MatrixXd::Zero(1, dim);
			for (i = 0; i < N; i++)
			{
				if (idx[i] == cluster)
				{
					counter++;
					tempMean += data.row(i);
				}
			}
			
			if (counter > 0)
			{
				center.row(cluster) = (tempMean / (double) counter);
			}
			else
			{
				// do not remove the cluster, randomly select one point as the new center
				center.row(cluster) = data.row(rand() % N);
			}
		}
	}
	delete [] idx;
}

/*! \fn void kMedoids(budgetedData *trainData, parameters *param, unsigned int *medoidIndex)
	\brief Selects the landmark points using k-medoids algorithm.
	\param [in] trainData Input training data.
	\param [in] param The parameters of the algorithm.
	\param [out] medoidIndex Indices of medoid centers.
	
	Selects the landmark points using k-medoids algorithm, may save memory as compared to k-means algorithm since if the training data is sparse the landmark points will be sparse as well.
	However, the algorithm is O(N^2), where N is number of data points loaded in the first data chunk, and it may be slower than k-means.
	
	Implemented according to Park, Hae-Sang, and Chi-Hyuck Jun. "A simple and fast algorithm for K-medoids clustering." Expert Systems with Applications 36.2 (2009): 3336-3341.
*/
void kMedoids(budgetedData *trainData, parameters *param, unsigned int *medoidIndex)
{
	unsigned int iter, cluster, N = trainData->N, i, j;
	double currentdist, totalDist, tmpTotalDist, minDist;
	
	// initialize the medoidIndex (random initialization)
	vector <unsigned int> randomPerm(N, 0);
	for (i = 0; i < N; i++)
		randomPerm[i] = i;
	random_shuffle(randomPerm.begin(), randomPerm.end());
	
	for (i = 0; i < param->BUDGET_SIZE; i++)
	{
		medoidIndex[i] = randomPerm[i];
	}
	
	vector <unsigned int> clusterIndex(N);
	for (iter = 0; iter < param->K_MEANS_ITERS; iter++)
	{
		// assign clusters first
		for (i = 0; i < N; i++)
		{
			//find the nearest medoid
			minDist = INF;
			for (cluster = 0; cluster < param->BUDGET_SIZE; cluster++)
			{
				currentdist = trainData->distanceBetweenTwoPoints(i, medoidIndex[cluster]);
				if (currentdist < minDist)
				{
					minDist = currentdist;
					clusterIndex[i] = cluster;
				}
			}
		}
		
		// update the medoids
		for (cluster = 0; cluster < param->BUDGET_SIZE; cluster++)
		{
			tmpTotalDist = INF;
			for (i = 0; i < N; i++)
			{
				if (clusterIndex[i] == cluster)
				{
					totalDist = 0.0;
					for (j = 0; j < N; j++)
					{
						if (clusterIndex[j] == cluster)
							totalDist += trainData->distanceBetweenTwoPoints(i, j);
					}
					if (totalDist < tmpTotalDist)
					{
						tmpTotalDist = totalDist;
						medoidIndex[cluster] = i; // I used maxIter as stop criteria (set to a small number). No need to store previous medoid index
					}
				}
			}
		}
	}	
}
/*! \fn void liblinear_Solve_l2r_l1(const MatrixXd &X, unsigned char *y, VectorXd &w, parameters *param, vector <int> *yLabels)
	\brief Solves linear C-SVM on the transformed data points, taken from LibLINEAR implementation.
	\param [in] X Transformed data which is to be solved by linear SVM, each row is one data point.
	\param [in] y Labels of the data points.
	\param [in,out] w Current linear SVM-model parameters modified by the function, zero-weight at the start of training.

	\param [in] param The parameters of the algorithm.
	
	The function computes a linear separating hyperplane between two classes. It is used after the data points are projected from their original input feature space to a new feature space, defined by the projection matrix \link modelLLSVMmatrixW\endlink. The function is taken nearly verbatim from the LibLINEAR package.
*/
void liblinear_Solve_l2r_l1(const MatrixXd &X, unsigned char *y, VectorXd &w, parameters *param, vector <int> *yLabels)
{
	// solve l2 regularized l1 loss SVM by dual coordinate descent method
	// min_\alpha   0.5(\alpha^T (Q) \alpha) - e^T \alpha
	//         s.t     0 <= alpha_i <= c
	//     where Qij = yi yj Xi^T Xj 
	
	// to train linear kernel we need -1 and +1 labels, but a user can give us any labels, e.g., 0/1 labels; therefore
	//	here we set the default labels, such that the first user-provided label is renamed as -1, and the second +1
	char defaultLabels[2] = {-1, 1};
	
	int l = (int) X.rows();      // number of examples
	int i, s, iter = 0;
	MatrixXd Q = MatrixXd::Zero(l,1);
	int maxIter = 30;
	int *index = Malloc(int, l);
	int activeSize = l;      // shrinkage
	double *alpha = Malloc(double, l);
	
	// PG: projected gradient, for shrinking and stopping
	double PG;
	double PGmaxOld = INF;
	double PGminOld = -INF;
	double PGmaxNew, PGminNew;

	for(i = 0; i < l; i++)
	{
		Q(i,0) = X.row(i)*(X.row(i).transpose());    // Q(i,i) = (x_i)^T*(x_i) for l1-loss svm
		index[i] = i;
		alpha[i] = 0;
	}

	while (iter < maxIter)
	{
		PGmaxNew = -INF;
		PGminNew = INF;

		for (i = 0; i < activeSize; i++)          	// randomly permute the examples in the active set
		{
			int j = i + rand() % (activeSize - i);
			mySwap(index[i], index[j]);
		}

		for (s = 0; s < activeSize; s++)
		{
			i = index[s];
			//double yi = (double) (*yLabels)[y[i]];
			double yi = (double) defaultLabels[y[i]];
			double tmp = w.dot(X.row(i));
			double G = tmp * yi - 1.0;				// gradient
			
			// projected gradient
			PG = 0.0;
			if (alpha[i] == 0.0)
			{
				if (G > PGmaxOld)
				{
					activeSize--;
					mySwap(index[s], index[activeSize]);
					s--;
					continue;
				}
				else if (G < 0.0)
				{	PG = G;  }
			}
			else if (alpha[i] == (*param).C_PARAM)
			{
				if (G < PGminOld)
				{
					activeSize--;
					mySwap(index[s], index[activeSize]);
					s--;
					continue;
				}
				else if(G > 0)
				{   PG = G; }
		
			}
			else
			{	PG = G;	}

			PGmaxNew = myMax(PGmaxNew, PG);
			PGminNew = myMin(PGminNew, PG);

			if (fabs(PG) > 1.0e-12)     // |PG| != 0
			{
				double alphaOld = alpha[i];
				alpha[i] = myMin(myMax(alpha[i] - G/(double)Q(i, 0), 0.0), (*param).C_PARAM);
				w = w + (alpha[i] - alphaOld)* yi * (X.row(i).transpose());					
			}
		}
		iter++;
		double eps = 0.01;      // stopping criterion
		if (PGmaxNew - PGminNew <= eps)
		{
			if (activeSize == l)
				break;
			else
			{
				activeSize = l;
				PGmaxOld = INF;
				PGminOld = -INF;
				continue;
			}
		}
		PGmaxOld = PGmaxNew;
		PGminOld = PGminNew;
		if (PGmaxOld <= 0)
			PGmaxOld = INF;
		if (PGminOld >= 0)
			PGminOld = -INF;
	}
	free(alpha);
	free(index);
	Q.resize(0, 0);
}

/* \fn float predictLLSVM(budgetedData *testData, parameters *param, budgetedModelLLSVM *model, vector <char> *labels)
	\brief Given an LLSVM model, predict the labels of testing data.
	\param [in] testData Input test data.
	\param [in] param The parameters of the algorithm.
	\param [in] model Trained LLSVM model.
	\param [out] labels Vector of predicted labels.
	\return Error rate.
*/
float predictLLSVM(budgetedData *testData, parameters *param, budgetedModelLLSVM *model, vector <char> *labels)
{
	// to train linear kernel we need -1 and +1 labels, but a user can give us any labels, e.g., 0/1 labels; therefor
	//	here we set the default labels, such that the first user-provided label is renamed as -1, and the second +1
	char defaultLabels[2] = {-1, 1};
	
    unsigned long N, err = 0, total = 0, timeCalc = 0, start;
	bool stillChunksLeft = true;
	char text[256];
	VectorXd v((*param).BUDGET_SIZE), temp((*param).BUDGET_SIZE);
	budgetedVectorLLSVM *currentData = NULL;
	long double tempSqrNorm;
	
	if ((*param).VERBOSE)
		svmPrintString("Computing lower-dimensional representation and predicting labels ...\n");

	while (stillChunksLeft)
	{ 
        stillChunksLeft = testData->readChunk((*param).CHUNK_SIZE);
		(*param).updateVerySparseDataParameter(testData->getSparsity());
		
        N = testData->N;
        total += N;
		start = clock();
		
		// calculate E, kernel between testing points and landmark points
    	VectorXd predictions(N);
		for (unsigned int i = 0; i < N; i++)
		{
			if ((*param).VERY_SPARSE_DATA)
			{
				// since we are computing kernels using vectors directly from the budgetedData, we need square norm of the vector to speed-up
				// 	computations, here we compute it just once; no need to do it in non-sparse case, since this norm can be retrieved directly
				// 	from budgetedVector
				
				tempSqrNorm = testData->getVectorSqrL2Norm(i, param);
				for (unsigned int j = 0; j < (*param).BUDGET_SIZE; j++)
				{
					v(j) = (double)(*(model->modelLLSVMlandmarks))[j]->gaussianKernel(i, testData, param, tempSqrNorm);
				}
			}
			else
			{
				// first create the budgetedVector using the vector from budgetedData, to be used in gaussianKernel() method below
				currentData = new budgetedVectorLLSVM((*param).DIMENSION, (*param).CHUNK_WEIGHT);
				currentData->createVectorUsingDataPoint(testData, i, param);
				
				for (unsigned int j = 0; j < (*param).BUDGET_SIZE; j++)
				{
					v(j) = (double)(*(model->modelLLSVMlandmarks))[j]->gaussianKernel(currentData, param);
				}
				delete currentData;
				currentData = NULL;
			}
			
			temp = v.transpose() * (*model).modelLLSVMmatrixW;
			predictions(i) = temp.dot((*model).modelLLSVMweightVector);
		}
		
		for (unsigned int i = 0; i < N; i++)
		{
			if ((predictions(i) > 0.0) != (defaultLabels[testData->al[i]] > 0))
    			err++;
			
			// save predicted label, will be sent to output
			if (labels)
				(*labels).push_back((char)(testData->yLabels)[(predictions(i) > 0)]);
		}
		
		timeCalc += clock() - start;
		
    	if (((*param).VERBOSE) && (N > 0))
    	{
			sprintf(text, "Number of examples processed: %ld\n", total);
			svmPrintString(text);
        }
    }
	
	if ((*param).VERBOSE)
    {
		sprintf(text, "*** Testing completed in %5.3f seconds\n*** Testing error rate: %3.2f percent", (double)timeCalc / (double)CLOCKS_PER_SEC, 100.0 * (float) err / (float) total);
		svmPrintString(text);
    }
	
    return (float) (100.0 * (float) err / (float) total);
}

/*! \fn selectLandmarkPoints(budgetedData *trainData, parameters *param, budgetedModelLLSVM *model)
	\brief Selects landmark points for LLSVM.
	\param [in] trainData The input data.
	\param [in] param The parameters of the algorithm.
	\param [out] model Model object, which keeps the selected landmark points.
*/
void selectLandmarkPoints(budgetedData *trainData, parameters *param, budgetedModelLLSVM *model)
{
	unsigned int N = trainData->N, i;
	budgetedVectorLLSVM *currentRow = NULL;
	vector <unsigned int> randomPerm;
	MatrixXd X, kMeansCenters;
	VectorXd tempVector;
	
	switch ((*param).MAINTENANCE_SAMPLING_STRATEGY)
	{
		case 0:
			// take random points as landmarks
			if ((*param).VERBOSE)
				svmPrintString("Taking random points as landmarks ...\n");				
			
			// get random permutation
			randomPerm.resize(N);
			for (i = 0; i < N; i++)
				randomPerm[i] = i;
			random_shuffle(randomPerm.begin(), randomPerm.end());
			
			// just take random data points as landmark points
			for (i = 0; i < (*param).BUDGET_SIZE; i++)
			{
				currentRow = new budgetedVectorLLSVM((*param).DIMENSION, (*param).CHUNK_WEIGHT);
				currentRow->createVectorUsingDataPoint(trainData, randomPerm[i], param);
				(*(model->modelLLSVMlandmarks)).push_back(currentRow);
				currentRow = NULL;
			}
			break;
		
		case 1:
			// run k-means to get landmarks
			if ((*param).VERBOSE)
				svmPrintString("Running k-means to choose landmark points ...\n");
			
			// perform k-means
			// first put the data into matrix (online k-means maybe?)
			X = MatrixXd::Zero(N, (*param).DIMENSION);
			
			for (i = 0; i < N; i++)
			{
				unsigned int istart = trainData->ai[i];    
				unsigned int iend = (i == trainData->N - 1) ? (unsigned int) trainData->aj.size() : trainData->ai[i + 1];
				for (unsigned int j = istart; j < iend; j++)
					X(i, trainData->aj[j] - 1) = trainData->an[j];
			}
			
			// perform k-means
			kMeansCenters = MatrixXd::Zero((*param).BUDGET_SIZE, (*param).DIMENSION);
			kMeans(X, (*param).BUDGET_SIZE, (*param).K_MEANS_ITERS, kMeansCenters);
			X.resize(0, 0);
			
			// insert landmark points as a matrix of data of size (points x dims)
			for (i = 0; i < (*param).BUDGET_SIZE; i++)
			{
				currentRow = new budgetedVectorLLSVM((*param).DIMENSION, (*param).CHUNK_WEIGHT);
				tempVector = kMeansCenters.row(i);
				currentRow->createVectorUsingDataPointMatrix(tempVector);
				(*(model->modelLLSVMlandmarks)).push_back(currentRow);
				currentRow = NULL;
			}			
			break;
			
		case 2:
			// run k-medoids to get landmarks
			if ((*param).VERBOSE)
				svmPrintString("Running k-medoids to choose landmark points ...\n");
			
			// perform k-medoid initialization
			unsigned int *centerIndex = new unsigned int[param->BUDGET_SIZE];
			kMedoids(trainData, param, centerIndex);
			if ((*param).VERBOSE)
				svmPrintString("K-medoids completed ...\n");
			
			for (i = 0; i < (*param).BUDGET_SIZE; i++)
			{
				currentRow = new budgetedVectorLLSVM((*param).DIMENSION, (*param).CHUNK_WEIGHT);
				currentRow->createVectorUsingDataPoint(trainData, centerIndex[i], param);
				(*(model->modelLLSVMlandmarks)).push_back(currentRow);
				currentRow = NULL;
			}
			delete [] centerIndex;
			centerIndex = NULL;			
			break;		
	}		
}

/* \fn void trainLLSVM(budgetedData *trainData, parameters *param, budgetedModel *model)
	\brief Train LLSVM.
	\param [in] trainData Input training data.
	\param [in] param The parameters of the algorithm.
	\param [in,out] model Initial LLSVM model.
	
	The function trains LLSVM model, given input data, the initial model (most often zero-weight model), and the parameters of the model.
*/
void trainLLSVM(budgetedData *trainData, parameters *param, budgetedModelLLSVM *model)
{
	unsigned long timeCalc = 0, start;
	unsigned int i, j, total = 0, N;
	bool stillChunksLeft = true, firstChunk = true;
	long double tempSqrNorm;
	char text[256];
	budgetedVectorLLSVM *currentData = NULL;
	
	// W matrix for Nystrom method, here employ Eigen library since we need complex matrix operations
	(*model).modelLLSVMmatrixW = MatrixXd::Zero((*param).BUDGET_SIZE, (*param).BUDGET_SIZE);
	
	// initialize weight (i.e., hyperplane) in the projected space to zero-vector
	(*model).modelLLSVMweightVector = VectorXd::Zero((*param).BUDGET_SIZE);
	
	// commence with LLSVM training procedure
	stillChunksLeft = true;
	while (stillChunksLeft)
	{			
		stillChunksLeft = trainData->readChunk((*param).CHUNK_SIZE);
		(*param).updateVerySparseDataParameter(trainData->getSparsity());
		
		if (trainData->yLabels.size() != 2)
		{
			sprintf(text, "LLSVM is a binary classifier, but %d class(es) detected!\n", (int) trainData->yLabels.size());
			svmPrintErrorString(text);
		}
		
		N = trainData->N;
		total += N;
		start = clock();
		
		// if we just started training initialize the landmark points
		if (firstChunk)
		{
			if (N < (*param).BUDGET_SIZE)
			{
				trainData->flushData();
				svmPrintErrorString("Number of landmark points larger than size of the loaded chunk!\n");
			}
			firstChunk = false;
			
			// select landmark points
			selectLandmarkPoints(trainData, param, model);
			
			if ((*param).VERBOSE)
				svmPrintString("Computing the mapping function ...\n");
				
			// compute the W matrix, done just once per training
			for (i = 0; i < (*param).BUDGET_SIZE; i++)
			{
				for (j = 0; j < (*param).BUDGET_SIZE; j++)
				{
					if (i <= j)
					{
						(*model).modelLLSVMmatrixW(i, j) = (double) (*(model->modelLLSVMlandmarks))[i]->gaussianKernel((*(model->modelLLSVMlandmarks))[j], param);
					}
					else
					{
						(*model).modelLLSVMmatrixW(i, j) = (*model).modelLLSVMmatrixW(j, i);
					}
				}
			}
			
			// finally, compute K_zz = W^(-0.5), initialization is complete
			invSquareRoot((*model).modelLLSVMmatrixW);
		}
		
		// done with initialization phase, next we compute the mapping in the new space and solve linear SVM
		
		if ((*param).VERBOSE)
			svmPrintString("Computing mapping of the training data ...\n");
		
		// here compute kernel matrix E between input data and landmark points
		MatrixXd E = MatrixXd::Zero(N, (*param).BUDGET_SIZE);
		for (i = 0; i < N; i++)
		{
			if ((*param).VERY_SPARSE_DATA)
			{
				// since we are computing kernels using vectors directly from the budgetedData, we need square norm of the vector to speed-up
				// 	computations, here we compute it just once; no need to do it in non-sparse case, since this norm can be retrieved directly
				// 	from budgetedVector
				tempSqrNorm = trainData->getVectorSqrL2Norm(i, param);
				for (j = 0; j < (*param).BUDGET_SIZE; j++)
				{
					E(i, j) = (double)(*(model->modelLLSVMlandmarks))[j]->gaussianKernel(i, trainData, param, tempSqrNorm);
				}	
			}
			else
			{
				// first create the budgetedVector using the vector from budgetedData, to be used in gaussianKernel() method below
				currentData = new budgetedVectorLLSVM((*param).DIMENSION, (*param).CHUNK_WEIGHT);
				currentData->createVectorUsingDataPoint(trainData, i, param);
				
				for (j = 0; j < (*param).BUDGET_SIZE; j++)
				{
					E(i, j) = (double)(*(model->modelLLSVMlandmarks))[j]->gaussianKernel(currentData, param);
				}			
				delete currentData;
				currentData = NULL;
			}
		}
		
		// compute new representation of data set which will be used to train SVM
		E = E * (*model).modelLLSVMmatrixW;		
		
		if ((*param).VERBOSE)
			svmPrintString("Training linear SVM ...\n");
		
		// now we can move on to training SVM using data in a new space
		liblinear_Solve_l2r_l1(E, trainData->al, (*model).modelLLSVMweightVector, param, &(trainData->yLabels));
		timeCalc += clock() - start;
		
		if (((*param).VERBOSE) && (N > 0))
		{
			sprintf(text, "Number of examples processed: %d\n", total);
			svmPrintString(text);
		}
	}	
	// training done, get rid of training data
	trainData->flushData();
	
	//timeCalc += clock() - startTotal;
    if ((*param).VERBOSE)
    {
		sprintf(text, "*** Training completed in %5.3f seconds.\n\n", (double)timeCalc / (double)CLOCKS_PER_SEC);
		svmPrintString(text);
    }
}

/* \fn bool budgetedModelLLSVM::saveToTextFile(const char *filename, vector <int>* yLabels, parameters *param)
	\brief Saves the trained model to .txt file.
	\param [in] filename Filename of the .txt file where the model is saved.
	\param [in] yLabels Vector of possible labels.
	\param [in] param The parameters of the algorithm.
	\return Returns false if error encountered, otherwise true.
*/
bool budgetedModelLLSVM::saveToTextFile(const char *filename, vector <int>* yLabels, parameters *param)
{
	unsigned int i, j;
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
	fprintf(fModel, " %d\n", (int) (*modelLLSVMlandmarks).size());
	
	// bias parameter
	fprintf(fModel, "BIAS_TERM: %f\n", (*param).BIAS_TERM);
	
	// kernel width (GAMMA) parameter
	fprintf(fModel, "KERNEL_WIDTH: %f\n", (*param).GAMMA_PARAM);
	
	// save the model
	fprintf(fModel, "MODEL:\n");
	for (i = 0; i < (*modelLLSVMlandmarks).size(); i++)
	{
		// put the i-th value of linear SVM hyperplane
		fprintf(fModel, "%2.10f", (double)modelLLSVMweightVector(i));
		
		// next, put the values of one row of modelLLSVMmatrixW
		for (j = 0; j < (*param).BUDGET_SIZE; j++)
			fprintf(fModel, " %2.10f", modelLLSVMmatrixW(i, j));
		
		// finally, store the landmark point
		for (j = 0; j < (*param).DIMENSION; j++)
		{
			if ((*((*modelLLSVMlandmarks)[i]))[j] != 0.0)
				fprintf(fModel, " %d:%2.10f", j + 1, (*((*modelLLSVMlandmarks)[i]))[j]);
		}
		fprintf(fModel, "\n");
	}
	
	fclose(fModel);
	return true;
}

/* \fn bool budgetedModelLLSVM::loadFromTextFile(const char *filename, vector <int>* yLabels, parameters *param)
	\brief Loads the trained model from .txt file.
	\param [in] filename Filename of the .txt file where the model is saved.
	\param [out] yLabels Vector of possible labels.
	\param [out] param The parameters of the algorithm.
	\return Returns false if error encountered, otherwise true.
*/
bool budgetedModelLLSVM::loadFromTextFile(const char *filename, vector <int>* yLabels, parameters *param)
{
	unsigned int i, numClasses;
	float tempFloat;
	string text;
	char oneWord[1024];
	int pos, tempInt;
	FILE *fModel = NULL;
	bool doneReadingBool;
	long double sqrNorm;
	
	fModel = fopen(filename, "rt");
	if (!fModel)
		return false;
	
	// algorithm
	fseek(fModel, strlen("ALGORITHM: "), SEEK_CUR);
	if (!fscanf(fModel, "%d\n", &((*param).ALGORITHM)))
	{
		svmPrintErrorString("Error reading algorithm type from the model file!\n");
	}
	
	// dimension
	fseek(fModel, strlen("DIMENSION:"), SEEK_CUR);
	if (!fscanf(fModel, "%d\n", &((*param).DIMENSION)))
	{
		svmPrintErrorString("Error reading number of dimensions from the model file!\n");
	}
	
	// number of classes (for LLSVM always equal to 2)
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
		svmPrintErrorString("Error reading number of weights from the model file!\n");
	}
	(*param).BUDGET_SIZE = tempInt;
	
	// bias parameter
	fseek (fModel, strlen("BIAS TERM: "), SEEK_CUR);
	if (!fscanf(fModel, "%f\n", &tempFloat))
	{
		svmPrintErrorString("Error reading bias term from the model file!\n");
	}
	(*param).BIAS_TERM = tempFloat;
	
	// kernel width (GAMMA) parameter
	fseek (fModel, strlen("KERNEL WIDTH: "), SEEK_CUR);
	if (!fscanf(fModel, "%f\n", &tempFloat))
	{
		svmPrintErrorString("Error reading kernel width from the model file!\n");
	}
	(*param).GAMMA_PARAM = tempFloat;
	
	// allocate memory for model
	modelLLSVMmatrixW.resize((*param).BUDGET_SIZE, (*param).BUDGET_SIZE);
	
	// initialize weight (i.e., hyperplane) in the projected space to zero-vector
	modelLLSVMweightVector.resize((*param).BUDGET_SIZE);
	
	// load the model
	fseek (fModel, strlen("MODEL:\n") + 1, SEEK_CUR);
	for (i = 0; i < (*param).BUDGET_SIZE; i++)
	{
		budgetedVectorLLSVM *eNew = new budgetedVectorLLSVM((*param).DIMENSION, (*param).CHUNK_WEIGHT);
		sqrNorm = 0.0L;
		
		// get alphas and features below
		
		// get linear SVM feature
		fgetWord(fModel, oneWord);
		modelLLSVMweightVector(i) = (double) atof(oneWord);
		
		// get elements of modelLLSVMmatrixW
		for (unsigned int j = 0; j < (*param).BUDGET_SIZE; j++)
		{
			fgetWord(fModel, oneWord);
			modelLLSVMmatrixW(i, j) = (double) atof(oneWord);
		}
		
		// get features
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
				(*eNew)[tempInt - 1] = tempFloat;
				sqrNorm += (long double)(tempFloat * tempFloat);
			}
		}
		eNew->setSqrL2norm(sqrNorm);
		
		(*modelLLSVMlandmarks).push_back(eNew);
		eNew = NULL;
	}
	
	fclose(fModel);
	return true;
}
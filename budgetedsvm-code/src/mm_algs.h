/*!
	\file mm_algs.h
	\brief Defines classes and functions used for training and testing of large-scale multi-hyperplane algorithms (AMM batch, AMM online, and Pegasos).
*/
/* 
	* This program is a free software; you can redistribute it and/or modify
	* it under the terms of the GNU General Public License as published by
	* the Free Software Foundation; either version 3 of the License, or
	* (at your option) any later version.
	
	Author	:	Nemanja Djuric
	Name	:	mm_algs.h
	Date	:	November 19th, 2012
	Desc.	:	Defines classes and functions used for training and testing of large-scale multi-hyperplane algorithms (AMM batch, AMM online, and Pegasos).
	Version	:	v1.01
*/

#ifndef _MM_ALGS_H
#define _MM_ALGS_H

#ifdef __cplusplus
extern "C" {
#endif

/*! \class budgetedVectorAMM
    \brief Class which holds sparse vector, which is split into a number of arrays to trade-off between speed of access and memory usage of sparse data, with added methods for AMM algorithms.
*/
class budgetedVectorAMM : public budgetedVector
{
	// friends so that they can set sqrL2norm property during model loading
	friend class budgetedModelAMM;
	friend class budgetedModelMatlabAMM;
	
	/*! \var long double degradation
		\brief Degradation of the vector.
		
		At each iteration during the training procedure of AMM algorithms and Pegasos all weights are degraded, meaning that their elements are pushed slightly towards 0. 
		This can, in addition to numerical issues, also be a problem when the dimensionality of the data set is large, as in naive implementation each feature needs to be degraded independently.
		However, instead of degrading each element separately, we can keep degradation level as a single number which is the same for all features, thus avoiding round-off problems and also speeding up the degradation step, which now amounts to a single multiplication operation. 
		
		Consequently, the actual feature value of a vector is equal to the value stored in \link array\endlink, multiplied by \link degradation\endlink.
	*/	
	protected:
        long double degradation;
		
	public:
    	/*! \fn budgetedVectorAMM(unsigned int  dim = 0, unsigned int  chnkWght = 0) : budgetedVector(dim, chnkWght)
			\brief Constructor, initializes the vector to all zeros, and also initializes \link degradation\endlink parameter.
			\param [in] dim Dimensionality of the vector.
			\param [in] chnkWght Size of each vector chunk.
		*/
		budgetedVectorAMM(unsigned int dim = 0, unsigned int  chnkWght = 0) : budgetedVector(dim, chnkWght)
		{
			degradation = 1.0;
		}
		
		/*! \fn double getSqrL2norm(void)
			\brief Returns \link sqrL2norm\endlink, a squared L2-norm of the vector, which accounts for the vector degradation.
			\return Squared L2-norm of the vector.
		*/
		long double getSqrL2norm(void)
		{
			return (degradation * degradation * sqrL2norm);
		}
		
		/*! \fn void downgrade(long oto)
			\brief Downgrade the existing weight-vector.
			\param [in] oto Total number of AMM training iterations so far.
			
			Using this function, each training iteration all non-zero weights are pushed closer to 0, to ensure the convergence of the algorithm to the optimal solution.
		*/
		void downgrade(long oto)
		{
			degradation *= (1.0 - 1.0 / ((long double)oto + 1.0));
		};
		
		/*! \fn long double sqrNorm(void)
			\brief Calculates a squared norm of the vector, but takes into consideration current degradation of a vector.
			\return Squared norm of the vector.
		*/	
		long double sqrNorm(void)
		{
			return (degradation * degradation * budgetedVector::sqrNorm());
		}
		
		/*! \fn long double getDegradation(void)
			\brief Returns \link degradation\endlink of a vector.
			\return Degradation of a vector.
		*/
		long double getDegradation(void)
		{
			return degradation;
		}
		
		/*! \fn void setDegradation(long double)
			\brief Sets \link degradation\endlink of a vector.
		*/
		void setDegradation(long double deg)
		{
			degradation = deg;
		}
		
		/*! \fn void updateDegradation(unsigned int iteration, parameters *param)
			\brief Computes \link degradation\endlink of a vector.
			\param [in] iteration Training iteration at which the degradation is set, used to compute the degradation value.
			\param [in] param The parameters of the algorithm.
		*/
		void updateDegradation(unsigned int iteration, parameters *param)
		{
			degradation = 1.0 / (((long double)iteration + 1.0) * (long double)(*param).LAMBDA_PARAM);
		}
		

		/*! \fn void updateUsingDataPoint(budgetedData* inputData, unsigned int oto, unsigned int t, int sign, parameters *param)
			\brief Updates a weight-vector when misclassification happens.
			\param [in] inputData Input data from which t-th vector is considered.
			\param [in] oto Total number of iterations so far.
			\param [in] t Index of the input vector in the input data.
			\param [in] sign +1 if the input vector is of the true class, -1 otherwise, specifies how the weights will be updated.
			\param [in] param The parameters of the algorithm.
			
			When we misclassify a data point during training, this function is used to update the existing weight-vector. It brings the true-class weight closer to the misclassified 
			data point, and to push the winning other-class weight away from the misclassified point according to AMM weight-update equations. The missclassified example used to update
			an existing weight is located in the input data set loaded to budgetedData.
		*/
		void updateUsingDataPoint(budgetedData* inputData, unsigned int oto, unsigned int t, int sign, parameters *param);
		

		/*! \fn void updateUsingVector(budgetedVectorAMM* otherVector, unsigned int oto, int sign, parameters *param)
			\brief Updates a weight-vector when misclassification happens.
			\param [in] otherVector Misclassified example used to update the existing weight.
			\param [in] oto Total number of iterations so far.
			\param [in] sign +1 if the input vector is of the true class, -1 otherwise, specifies how the weights will be updated.
			\param [in] param The parameters of the algorithm.
			
			When we misclassify a data point during training, this function is used to update the existing weight-vector. It brings the true-class weight closer to the misclassified 
			data point, and to push the winning other-class weight away from the misclassified point according to AMM weight-update equations. The missclassified example used to update
			an existing weight is located in the budgetedVectorAMM object.
		*/
		void updateUsingVector(budgetedVectorAMM* otherVector, unsigned int oto, int sign, parameters *param);
		

		/*! \fn void createVectorUsingDataPoint(budgetedData* inputData, unsigned int oto, unsigned int t, parameters *param)
			\brief Create new weight from one of the zero-weights.
			\param [in] inputData Input data from which t-th vector is considered.
			\param [in] oto Total number of iterations so far.
			\param [in] t Index of the input vector in the input data.
			\param [in] param The parameters of the algorithm.
			
			The function simply copies the t-th data point in the input data to the vector vij, while also updating the degradation variable.
		*/
		void createVectorUsingDataPoint(budgetedData* inputData, unsigned int oto, unsigned int t, parameters *param)
		{
			budgetedVector::createVectorUsingDataPoint(inputData, t, param);    
			degradation = 1.0 / (((long double)oto + 1.0) * (long double)(*param).LAMBDA_PARAM);
		}
		
		/*! \fn virtual long double linearKernel(unsigned int t, budgetedData* inputData, parameters *param)
			\brief Computes linear kernel between vector and given input data point, but also accounts for degradation.
			\param [in] t Index of the input vector in the input data.
			\param [in] inputData Input data from which t-th vector is considered.
			\param [in] param The parameters of the algorithm.
			\return Value of linear kernel between two input vectors.
			
			Function computes the dot product (i.e., linear kernel) between budgetedVector vector and the input data point from budgetedData. 
		*/
		long double linearKernel(unsigned int t, budgetedData* inputData, parameters *param)
		{
			return (degradation * budgetedVector::linearKernel(t, inputData, param));
		};
		
		/*! \fn virtual long double linearKernel(budgetedVectorAMM* otherVector)
			\brief Computes linear kernel between this budgetedVectorAMM vector and another vector stored in budgetedVectorAMM, but also accounts for degradation.
			\param [in] otherVector The second input vector to linear kernel.
			\return Value of linear kernel between two input vectors.
			
			Function computes the dot product (or linear kernel) between two vectors.
		*/
		long double linearKernel(budgetedVectorAMM *otherVector)
		{
			return (degradation * otherVector->getDegradation() * budgetedVector::linearKernel(otherVector));
		};
};

/*!
    \brief A vector of vectors, implements the weight matrix of AMM algorithms as jagged array.
*/
typedef vector <budgetedVectorAMM*> vectorOfBudgetVectors;

/*! \class budgetedModelAMM
    \brief Class which holds the AMM model, and implements methods to load AMM model from and save AMM model to text file.
*/
class budgetedModelAMM : public budgetedModel
{
	/*! \var vector <vectorOfBudgetVectors> *modelMM
	\brief Holds AMM batch, AMM online, or PEGASOS models.
	*/	
	protected:
		vector <vectorOfBudgetVectors> *modelMM;
		
	public:		
		/*! \fn budgetedModelAMM(void)
			\brief Constructor, initializes the MM model to zero weights.
		*/
		budgetedModelAMM(void)
		{
			modelMM = new vector <vectorOfBudgetVectors>;
		};
		
		/*! \fn ~budgetedModelAMM(void)
			\brief Destructor, cleans up memory taken by AMM.
		*/	
		~budgetedModelAMM(void);
		
		/*! \fn vector <vectorOfBudgetVectors> * getModel(void)
			\brief Used to obtain a pointer to a current AMM model.
			\return A pointer to a current AMM model.
		*/
		vector <vectorOfBudgetVectors> * getModel(void)
		{
			return modelMM;
		}
		
		/*! \fn bool saveToTextFile(const char *filename, vector <int>* yLabels, parameters *param)
			\brief Saves the trained AMM model to .txt file.
			\param [in] filename Filename of the .txt file where the model is saved.
			\param [in] yLabels Vector of possible labels.
			\param [in] param The parameters of the algorithm.
			\return Returns false if error encountered, otherwise true.
			
			The text file has the following rows: [\a ALGORITHM, \a DIMENSION, \a NUMBER_OF_CLASSES, \a LABELS, \a NUMBER_OF_WEIGHTS, \a BIAS_TERM, \a KERNEL_WIDTH, \a MODEL]. In order to compress memory and to use the 
			memory efficiently, we coded the model in the following way:
			
			Each row of the text file corresponds to one weight. The first element of each weight is the class of the weight, followed by the degradation of the weight. 
			The rest of the row corresponds to non-zero elements of the weight, given as feature_index:feature_value, in a standard LIBSVM format.
		*/
		bool saveToTextFile(const char *filename, vector <int>* yLabels, parameters *param);
		
		/*! \fn bool loadFromTextFile(const char *filename, vector <int>* yLabels, parameters *param)
			\brief Loads the trained AMM model from .txt file.
			\param [in] filename Filename of the .txt file where the model is saved.
			\param [out] yLabels Vector of possible labels.
			\param [out] param The parameters of the algorithm.
			\return Returns false if error encountered, otherwise true.
			
			The text file has the following rows: [\a ALGORITHM, \a DIMENSION, \a NUMBER_OF_CLASSES, \a LABELS, \a NUMBER_OF_WEIGHTS, \a BIAS_TERM, \a KERNEL_WIDTH, \a MODEL]. In order to compress memory and to use the 
			memory efficiently, we coded the model in the following way:
			
			Each row of the text file corresponds to one weight. The first element of each weight is the class of the weight, followed by the degradation of the weight. 
			The rest of the row corresponds to non-zero elements of the weight, given as feature_index:feature_value, in a standard LIBSVM format.
		*/
		bool loadFromTextFile(const char *filename, vector <int>* yLabels, parameters *param);
};


/*! \fn void trainPegasos(budgetedData *trainData, parameters *param, budgetedModelAMM *model)
	\brief Train Pegasos.
	\param [in] trainData Input training data.
	\param [in] param The parameters of the algorithm.
	\param [in,out] model Initial Pegasos model.
	
	The function trains Pegasos model, given input data, initial model (most often zero-weight model), and the parameters of the model.
*/
void trainPegasos(budgetedData *trainData, parameters *param, budgetedModelAMM *model);

/*! \fn void trainAMMonline(budgetedData *trainData, parameters *param, budgetedModelAMM *model)
	\brief Train AMM online.
	\param [in] trainData Input training data.
	\param [in] param The parameters of the algorithm.
	\param [in,out] model Initial AMM model.
	
	The function trains multi-hyperplane machine using AMM online algotihm, given input data, the initial model (most often zero-weight model), and the parameters of the model.
*/
void trainAMMonline(budgetedData *trainData, parameters *param, budgetedModelAMM *model);

/*! \fn void trainAMMbatch(budgetedData *trainData, parameters *param, budgetedModelAMM *model)
	\brief Train AMM batch.
	\param [in] trainData Input training data.
	\param [in] param The parameters of the algorithm.
	\param [in,out] model Initial AMM model.
	
	The function trains multi-hyperplane machine using AMM batch algotihm, given input data, the initial model (most often zero-weight model), and the parameters of the model.
*/
void trainAMMbatch(budgetedData *trainData, parameters *param, budgetedModelAMM *model);


/*! \fn float predictAMM(budgetedData *testData, parameters *param, budgetedModelAMM *model, vector <char> *labels)
	\brief Given a multi-hyperplane machine (MM) model, predict the labels of testing data.
	\param [in] testData Input test data.
	\param [in] param The parameters of the algorithm.
	\param [in] model Trained MM model.
	\param [out] labels Vector of predicted labels.
	\return Testing set error rate.
	
	Given the learned multi-hyperplane machine, the function computes the predictions on the testing data, outputing the predicted labels and the error rate.
*/
float predictAMM(budgetedData *testData, parameters *param, budgetedModelAMM *model, vector <char> *labels);

#ifdef __cplusplus
}
#endif

#endif /* _MM_ALGS_H */
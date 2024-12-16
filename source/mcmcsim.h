#ifndef MCMC_H_
#define MCMC_H_
#include<stdio.h>
#include<stdlib.h>
#include<gsl/gsl_matrix.h>
#include<gsl/gsl_blas.h>
#include<gsl/gsl_vector.h>
#include <gsl/gsl_linalg.h>
#include<gsl/gsl_rng.h>
#include<gsl/gsl_randist.h>
#include<gsl/gsl_statistics_double.h>
#include<time.h>
#include<math.h>

/*
	****************************************************************************************************************
	 * Description:

	 * This function transforms the .txt data into gsl_matrices and calls the sampler. Two .bin files are saved as
	 * output, one containing the sampled chain and the other containing the log posterior calculated at every element
	 * of the chain.
	  ---------------------------------------------------------------------------------------------------------------
	 * Parameters:
	  
	 * FILE *data: a .txt file containing the data and the independent variable;

	 * FILE *cov: a .txt file containing the covariance matrix of the data;

	 * gsl_vector *model (gsl_vector*, gsl_vector*): the theoretical model. This function must take as input the independent 
	 * variable (which is now a gsl_vector) and a gsl_vector which will contains the sample to be tested. The output must be
	 * a gsl_vector of the size of the data containing the model function calculated with sampled value at every independent
	 * variable value following the order of the .txt file;
	 * 
	 * double prior: a function that calculates the log prior probability at the values sampled. The input is the gsl_vector
	 * containing the sampled point and the output is the log prior calculated at this point;
	 * 
	 * gsl_vector *first_element: the first element of the chain;
	 * 
	 * gsl_vector *proposal_density: the distribuition of which the values will be sampled. The argument of this function
	 * is the last element of the chain;
	 * 
	 * size_t n_data: the amount of data points;
	 * 
	 * size_t n_param: the number of parameters to be estimated;
	 * 
	 * size_t size: the number of elements in the chain.
	 * 
	 * int seed: the seed of the random number generator;
	 * 
	 * char *output_chain: the name of the output file that will contain the chain;
	 * 
	 * char *output_post: the name of the output file that will contain the log posterior calculated at every element of the chain.
	 
	 ***********************************************************************************************************************
*/

int mcmc(FILE *data, 
	FILE *cov,
	gsl_vector *model(gsl_vector*, gsl_vector*), 
	double prior(gsl_vector*),
	gsl_vector *first_element, 
	gsl_vector *proposal_density(gsl_vector *),
	size_t n_data,  
	size_t n_param,  
	size_t size,
	int seed,
	char *output_chain,
	char *output_post);

/*
	 **************************************************************************************************************
	 *Description:

	 *  This function calculates the log posterior probability given the sample.
	 --------------------------------------------------------------------------------------------------------------
	 * Parameters:
	 
	 * gsl_vector *x: the gsl_vector that contains the independent variable;

	 * gsl_vector *y: the gsl_vector that contains the data;

	 * gsl_matrix *data_cov_matrix_decomposed: the matrix that contains the lower triangular matrix L and the diagonal matrix D
	 * from the decomposition of the covariance matrix of the data;
	 
	 * gsl_vector *diagonal: the vector that contains the diagonal matrix product of the decomposition of the data covariance matrix;
	 
	 * gsl_vector *params: the vector that contains the sampled parameters of the theory to be tested;
*/

double logpost(gsl_vector *x, 
	gsl_vector *y, 
	gsl_matrix *data_cov_matrix_decomposed,
	gsl_vector *diagonal, 
	gsl_vector *model(gsl_vector*, gsl_vector*), 
	double prior(gsl_vector*), 
	gsl_vector *params, 
	size_t n_data);


/*
	 **********************************************************************************************************
	 *Description:
	 
	 * Sampler.

	 **********************************************************************************************************
	 */
int mh(gsl_vector *x, 
	gsl_vector*y, 
	gsl_matrix *data_cov_matrix_decomposed, 
	gsl_vector *diagonal, 
	gsl_vector *first_element,
	gsl_vector *model(gsl_vector*, gsl_vector*),
	double prior (gsl_vector*),
	gsl_vector *proposal_density (gsl_vector *),
	size_t n_data, 
	size_t n_param, 
	size_t size, 
	gsl_rng *r,
	char *output_chain,
	char *output_post);

/*
	 *****************************************************************************************************
	 * Description:

	 * This functions decides whether the proposed element will be accepted in the chain or not.

	 -----------------------------------------------------------------------------------------------------
	 * Parameters:
	 
	 * double ratio: the ratio between the posterior calculated at the proposed elements by the posterior calculated at the last
	 * element of the chain.

	 * gsl_rng *r: the random number generator.
	
	 *****************************************************************************************************
	 */

int decision (double ratio, 
	gsl_rng *r);

#endif 

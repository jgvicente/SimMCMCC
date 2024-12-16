#include "mcmcsim.h"

/* This file contains a  simple implementations Metropolis-Hastings algorithm.
* Author: João Gabriel Vicente.
* Date: September,2022.
*/

/* This function is called by the user*/

int mcmc(FILE *data, // File containing the data
	 	 FILE *cov, // File containing the covariance matrix of the data
		 gsl_vector *model(gsl_vector*, gsl_vector*), // Function that calculates the model
		 double prior(gsl_vector*), // Function that calculated the prior
		 gsl_vector *first_element, // First element of the Markov chain
		 gsl_vector *proposal_density(gsl_vector *), // Function that calculates the proposal density
		 size_t n_data, // Number of data points
		 size_t n_param, // Number of parameters
		 size_t size, // Size of the chain
		 int seed, // Seed for the random number generator
		 char *output_chain, // Name of the output .bin file that will contain the Markov chain
		 char *output_post // Name of the output .bin file that will contain the log posterior
		 )
{
	
	/*converting the .txt data and covariance matrix into gsl_matrix and gsl_vector. The covariance matrix will be decomposed
	into LDLT.*/
	gsl_matrix *data_array = gsl_matrix_alloc(n_data, 2);
	gsl_vector_view y;
	gsl_vector_view x;
	gsl_matrix *data_cov_matrix = gsl_matrix_alloc(n_data,n_data);
	gsl_vector_view diag; //the vector that will point to the diagonal part of the covariance matrix decomposition.

	/* Obtaining the data from the file*/
	gsl_matrix_fscanf(data, data_array);
	x = gsl_matrix_column(data_array, 0);
	y = gsl_matrix_column(data_array, 1);

	/* Obtaining the covariance matrix from the file*/
	gsl_matrix_fscanf(cov, data_cov_matrix);
	gsl_linalg_ldlt_decomp(data_cov_matrix); //decompose the covariance matrix into LDLT
	diag = gsl_matrix_diagonal (data_cov_matrix);

	/*Setting the random number generator*/
	gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937); 
	gsl_rng_set(r, seed);

	/*Calling the sampler */
	mh(&x.vector, &y.vector,data_cov_matrix, &diag.vector,first_element,model, prior,proposal_density,n_data, n_param, size, r, output_chain,
		output_post);

	gsl_matrix_free(data_array);
  	gsl_matrix_free(data_cov_matrix);
  	gsl_rng_free(r);
  	return 0;
}

/*
	 **************************************************************************************************************
	 *Procedure used to calculate the log likelihood:

	 * Let u and x be two vectors in the parameter space. The Mahalanobis distance is

	                                                   r = (u - x)^T C^{-1} (u - x) = v^T C^{-1} v,

	 * where C is the covariance matrix, v = u - x and v^T is the transposed matrix. Since C is a symmetrical semi-positive 
	 * definite function, it can be decomposed into a lower triangular matrix L and a diagonal matrix D in the form

	                                                   C = LDL^T.

	 * Therefore, taking w = L^{-1} v, the Mahalanobis distance can be written as

	                                                   r = w^T D^{-1} w.

	 * To find r, w must be determined. Since L is a triangular matrix, the computation of w is generaly fast.

	 ***************************************************************************************************************
	 */
double logpost(gsl_vector *x, // Vector of independent variable
			   gsl_vector *y, // Vector containing the data
			   gsl_matrix *data_cov_matrix_decomposed, // Matrix containing the lower triangular matrix L and the diagonal matrix D
			   gsl_vector *diagonal, // Vector containing the diagonal matrix product of the decomposition of the data covariance matrix
			   gsl_vector *model(gsl_vector*, gsl_vector*), // Function that calculates the model
			   double prior(gsl_vector*), // Function that calculates the prior
			   gsl_vector *params, // Vector containing the sampled parameters of the theory to be tested
			   size_t n_data // Number of data points
			   )
{

	/*Auxiliary variables. They are defined because the function gsl_vector_sub alters one of the vectors.*/
	gsl_vector *y_h = gsl_vector_alloc(n_data);
	gsl_vector *y_hh = gsl_vector_alloc(n_data);

	gsl_vector *func_calculated = gsl_vector_alloc(n_data);
	double llikelihood;
	double prior_calculated;
	gsl_blas_dcopy(y, y_h); // Copying the data vector to y_h

	//(mu - func)
	func_calculated = model(x, params);
	gsl_vector_sub(y_h, func_calculated);

	gsl_blas_dtrsv(CblasLower, CblasNoTrans, CblasUnit, data_cov_matrix_decomposed, y_h); //w
	gsl_blas_dcopy(y_h,y_hh);

	gsl_vector_div(y_hh, diagonal); //D^{-1} w
	gsl_blas_ddot(y_h, y_hh, &llikelihood); //-2log(Like)
	prior_calculated = prior(params);

	gsl_vector_free(func_calculated);
	gsl_vector_free(y_h);
	gsl_vector_free(y_hh);

	return prior_calculated -llikelihood/2;
	
}

int mh(gsl_vector *x, 
	   gsl_vector *y, 
	   gsl_matrix *data_cov_matrix_decomposed, 
	   gsl_vector *diagonal, gsl_vector *first_element, 
	   gsl_vector *model(gsl_vector*, gsl_vector*),
	   double prior (gsl_vector*),
	   gsl_vector *proposal_density (gsl_vector *),
	   size_t n_data, 
	   size_t n_param, 
	   size_t size,
	    gsl_rng *r, 
		char *output_chain, 
		char *output_post)
{

	gsl_matrix *chain = gsl_matrix_alloc(size,n_param); // markov chain
	double post[size]; // an array that will save the log posterior
	int accepted = 0; //acceptance count

	/*Setting the chain and the posterior to the first element */
	gsl_matrix_set_row(chain, 0, first_element);
	gsl_vector_view chain_row = gsl_matrix_row (chain, 0);
	post[0] = logpost(x, y, data_cov_matrix_decomposed, diagonal,model, prior, &chain_row.vector, n_data);
	
	/*Setting the next elements*/
	for(int i=1; i<size; i++)
	{
		gsl_vector *chain_test = gsl_vector_alloc(n_param); //vector that will contain the proposed element
		double post_test; //the log posterior calculated at the proposed element
		double ln_ratio;
		double ratio;
		int dec;
		gsl_vector_view param_row;

		param_row = gsl_matrix_row (chain, i-1);
		chain_test = proposal_density(&param_row.vector);
		post_test = logpost (x,y,data_cov_matrix_decomposed,diagonal,model, prior, chain_test, n_data);

		ln_ratio =   post_test - post[i-1];
		ratio = exp(ln_ratio);
		dec = decision(ratio, r);

		if (dec == 0)
		{
			post[i] = post_test;
			gsl_matrix_set_row(chain, i, chain_test);
			accepted++;
		}
		else
		{
			post[i] = post[i-1];
			gsl_matrix_set_row (chain, i, &param_row.vector);
		}

		gsl_vector_free(chain_test);
	}

	printf("acceptance rate = %f\n", (float) accepted / size);

	FILE *fpp = fopen(output_chain, "wb");
    gsl_matrix_fwrite(fpp, chain);
    FILE *fppp = fopen(output_post, "wb");
    fwrite(post, sizeof(double), size, fppp);

    fclose(fpp);
    fclose(fppp);
    gsl_matrix_free (chain);

    return 0;

}

int decision(double ratio, gsl_rng *r)
{

	if (ratio >= 1.0)
		{
			return 0;
		}

		else
		{
			double prob_test = gsl_rng_uniform(r);

			if(ratio >= prob_test)
			{
				return 0;
			}

			else
			{
				return 1;
			}

		}
}

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
#include "../source/mcmcsim.h"

/* Global Variables*/
size_t n_param; // Number of parameters
size_t n_data; // Number of data points
size_t size; // Size of the chain

gsl_matrix *fisher; // Fisher matrix for the prior
gsl_matrix *param_cov_matrix; // Covariance matrix for the proposal density
gsl_rng *r2; // Random number generator

/* Theoretical model
 * x: Vector of dependend variables
 * params: parameters of the model
*/
gsl_vector *model(gsl_vector *x, gsl_vector *params);

/* Prior distribution
 * params: parameters of the model
*/
double prior (gsl_vector * params);

/* Proposal density
 * last_element: last element of the chain
*/
gsl_vector *proposal_density (gsl_vector *last_element);

int main()
{
	clock_t begin = clock();
	n_data = 10;
	n_param = 3;
	size = 200000;
	/* Getting the data*/
	FILE *data =fopen("./data/data3p.txt", "r");
	FILE *cov = fopen("./data/data_cov_matrix3p.txt", "r");

	/* Getting the Fisher matrix and the covariance matrix for the user-defined functions*/
	FILE *fish = fopen("./data/fisher3p.txt", "r");
	FILE *cov_prop = fopen("./data/fisher_inv3p.txt", "r"); // Covariance matrix for the proposal density
	fisher = gsl_matrix_alloc(n_param,n_param);
	param_cov_matrix = gsl_matrix_alloc(n_param,n_param);
	gsl_matrix_fscanf(fish,fisher);
	gsl_matrix_fscanf(cov_prop, param_cov_matrix);

	gsl_linalg_cholesky_decomp1(param_cov_matrix); // Cholesky decomposition of the covariance matrix

	/* Random number generator*/
	r2= gsl_rng_alloc(gsl_rng_mt19937); 
	gsl_rng_set(r2, 1336);

	/* Setting the first element of the Markov Chain*/
	gsl_vector *first_element = gsl_vector_alloc(n_param);
	gsl_vector_set(first_element,0, 0);
	gsl_vector_set(first_element,1, 0);
	gsl_vector_set(first_element, 2, 0);

	/* Setting the seed for the sampler*/
	int seed = 1337;

	/* Running the MCMC*/
	mcmc(data, cov, model, prior, first_element, proposal_density, n_data, n_param, size, seed, 
	"./output/chain.bin", "./output/posterior.bin");

	fclose(data);
	fclose(cov);
	fclose(fish);
	fclose(cov_prop);
	gsl_vector_free(first_element);
	gsl_matrix_free(fisher);
	gsl_matrix_free(param_cov_matrix);
	gsl_rng_free (r2);

	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	printf("Time: %f\n", time_spent);
	return 0;
}

gsl_vector *model(gsl_vector *x, gsl_vector *params)
{
	gsl_vector *func = gsl_vector_alloc (n_data);
	double model;
	for(int i =0; i<n_data; i++)
	{
		model = gsl_vector_get(params,0) + gsl_vector_get(params, 1) * gsl_vector_get(x, i) 
		      + gsl_vector_get(params, 2) * gsl_pow_2( gsl_vector_get (x,i));
		gsl_vector_set (func, i, model);
	}
	return func;
}

double prior (gsl_vector *params)
{
	gsl_vector *product = gsl_vector_alloc(n_param);
	double pprior;
	/* (log of a ) Gaussian distribution with mean (0,0,0) and inverce variance given by the Fisher matrix*/
	gsl_blas_dgemv(CblasNoTrans, 1.0, fisher, params, 0.0, product);
	gsl_blas_ddot(params, product, &pprior);

	pprior = 0; // Uniform distribution
	gsl_vector_free (product);
	return -pprior/2;
}



gsl_vector *proposal_density(gsl_vector *last_element)
{
	gsl_vector *random_vector = gsl_vector_alloc(n_param);

	gsl_ran_multivariate_gaussian(r2, last_element, param_cov_matrix, random_vector);
	//printf("%f\n", gsl_vector_get(random_vector,0));
	return random_vector;

}


# Simple Markov Chain Monte Carlo Algorithm in C (SimMCMCC)

> A basic implementation of the Metropolis-Hastings Algorithm.

## About

This is a practice project whose goals is to explore:

1. The basics of the C programming language.
2. Markov Chain Monte Carlo (MCMC) methods, particularly the Metropolis-Hastings Algorithm.

## Installation

To build the program:

1. Open a command-line in the source folder.
2. Run the following command:

   ```bash
   make
   ```

If your compiler is not `gcc`, modify the `makefile` accordingly.

- Requirements

1. [GSL (GNU Scientific Library)](https://www.gnu.org/software/gsl/);

2. [GetDist](https://getdist.readthedocs.io/en/latest/) (used in the example notebook).

## Usage

For a quick start, check out the `example` folder. Below are the basic steps to use this library:

1. Create a .c file (main.c, for example) that defines the statistical model and loads the data and the covariance matrix.

- In the main function:

  - Load the data:

  ```C
  FILE *data =fopen("/pathtodata/data.txt", "r");
  FILE *cov = fopen("/pathtodata/data_cov_matrix.txt", "r");
  ```

  - If the log-prior or the proposal density require covariance matrices, load them as well

  ```C
  FILE *cov_prior =fopen("/pathtodata/cov_prior.txt", "r");
  FILE *cov_proposal = fopen("/pathtodata/cov_proposal.txt", "r");
  ```

  - Call the mcmc function. For a detailed explanation of the arguments of this function, see the `mcmcsim.h` file.

- Outside the main function:

  - Define the log-prior function, which return the log-prior calculated at params in the parameter space

  ```C
  double prior (gsl_vector * params);
  ```

  - Define the proposal density, which generates a proposed point in the parameter space based on the last element of the Markov Chain.

  ```C
  gsl_vector *proposal_density (gsl_vector *last_element);
  ```

  > Note: use the GSL routines to define these functions.

  - To run the MCMC, type in the command-line:

  ```bash
  gcc main.c -L../source -o omain -lmcmcsim `gsl-config --cflags --libs`
  ```

  Then run

  ```bash
  ./omain
  ```

## Example: three parameters linear model

The `smcmc_example.ipynb` file contains a detailed example.

- Model input: $\theta = [-0.51, 0.37, 0.084]$, $\epsilon \sim \mathcal{N}(0, 0.1)$.

- Result from SimMCMCC: $\bar{\theta} = [-0.50, 0.39, 0.088] \pm [0.05, 0.01, 0.006]$.

- Outputs:

  1.Corner Plot comparing the Markov Chain samples (red) with the Fisher matrix centered at the input values (blue)

  ![Corner Plot](/example/output/corner_plot.jpg)

  2.Convergence of the Markov Chain
  
  ![Convergence](/example/output/convergence.jpg)

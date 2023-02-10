#include <stdlib.h>
#include <time.h>
#include "utils.h"

// Stochastic gradient descent
void dsgd( size_t m, size_t n, const double *X, const double *y, double *coef, double *intercept, double eta, int max_iter, int fit_intercept, int random_seed )
{
    int n_odd = n % 2;

    double *gradient_coef = calloc (n, sizeof(double));
    
    double resid;
    //double alpha=1.0, beta=0.0;
    size_t idx;
    MTRand seed;
    double y_hat;

    if (random_seed < 0)
    {
        seed = seedRand( time(NULL) );
    }
    else
    {
        seed = seedRand( random_seed );
    }
    
    for ( int epoch = 0; epoch < max_iter; epoch++ )
    {
        for ( size_t run = 0; run < m; run++ )
        {
            // Randomly sample an observation
            idx = genRandLong(&seed) % m;
            
            //y_hat = dsigmoid( n, alpha, &X[n*idx], coef, beta, *intercept );
            y_hat = dsigmoid( n, &X[n*idx], coef, *intercept, n_odd );
            resid = -(y[idx] - y_hat);
            // Compute gradients and adjust weights
            for ( size_t i = 0; i < n; i++ )
            {
                gradient_coef[i] = X[n*idx + i] * resid;
                coef[i] -= eta * gradient_coef[i];
            }
            if ( fit_intercept == 1 )
            {
                *intercept -= eta * resid;
            }
        }
    }
    free(gradient_coef);
}

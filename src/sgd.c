#include <math.h>
#include <time.h>
#include "utils.h"
//#include "blis.h"

// Stochastic gradient descent
void dsgd( unsigned long m, unsigned long n, double *X, double *y, double *coef, double *intercept, double eta, int max_iter, int fit_intercept, int random_seed )
{
    double *gradient_coef;
    double y_pred, resid;
    unsigned long idx;
    MTRand seed;
    double alpha = 1.0, beta = 1.0;

    gradient_coef = (double *) malloc (n * sizeof(double));
    
    for ( unsigned long i = 0; i < n; i++ )
    {
        coef[i] = 0.0;
        gradient_coef[i] = 0.0;
    }
    *intercept = 0.0;
    

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
        for ( unsigned long run = 0; run < m; run++ )
        {
            // Randomly sample an observation
            idx = genRandLong(&seed) % m;
            
            // Compute y_hat
            y_pred = dsigmoid( n, alpha, &X[n*idx], coef, beta, *intercept );
            resid = -(y[idx] - y_pred);
            // Compute gradients and adjust weights
            for ( unsigned long i = 0; i < n; i++ )
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
}

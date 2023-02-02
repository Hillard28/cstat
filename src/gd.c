#include <stdlib.h>
#include "utils.h"

// Gradient descent
void dgd( unsigned long m, unsigned long n, double *X, double *y, double *coef, double *intercept, double eta, int max_iter, int fit_intercept )
{
    double alpha = 1.0, beta = 0.0, gradient_intercept = 0.0;
    double eta_m = eta / (double)m;

    double *resid = (double *) malloc (m * sizeof(double));
    double *gradient_coef = (double *) malloc (n * sizeof(double));

    for ( unsigned long j = 0; j < n; j++ )
    {
        coef[j] = 0.0;
        gradient_coef[j] = 0.0;
    }
    *intercept = 0.0;
    
    for ( int epoch = 0; epoch < max_iter; epoch++ )
    {
        // Compute y_hat and gradients
        for ( unsigned long i = 0; i < m; i++ )
        {
            resid[i] = y[i] - dsigmoid( n, alpha, &X[n*i], coef, beta, *intercept );
            for ( unsigned long j = 0; j < n; j++ )
            {
                gradient_coef[j] -= (X[n*i + j] * resid[i]);
            }
            if ( fit_intercept == 1 )
            {
                gradient_intercept -= resid[i];
            }
        }
        // Adjust weights
        for ( unsigned long j = 0; j < n; j++ )
        {
            coef[j] -= eta_m * gradient_coef[j];
            gradient_coef[j] = 0.0;
        }
        if ( fit_intercept == 1 )
        {
            *intercept -= eta_m * gradient_intercept;
            gradient_intercept = 0.0;
        }
    }
    free(resid);
    free(gradient_coef);
}

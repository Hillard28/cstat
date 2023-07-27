#include "utils.h"
//#include "blis.h"

// Stochastic gradient descent
void dgd( unsigned long m, unsigned long n, double *X, double *y, double *coef, double *intercept, double eta, int max_iter, int fit_intercept )
{
    double *gradient_coef, *y_pred, *resid;
    double alpha = 1.0, beta = 1.0, gradient_intercept = 0.0;

    gradient_coef = (double *) malloc (n * sizeof(double));
    y_pred = (double *) malloc (m * sizeof(double));
    resid = (double *) malloc (m * sizeof(double));

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
            y_pred[i] = dsigmoid( n, alpha, &X[n*i], coef, beta, *intercept );
            resid[i] = -(y[i] - y_pred[i]);
            for ( unsigned long j = 0; j < n; j++ )
            {
                gradient_coef[j] += X[n*i + j] * resid[i] / (double)m;
            }
            if ( fit_intercept == 1 )
            {
                gradient_intercept += resid[i] / (double)m;
            }
        }
        // Adjust weights
        for ( unsigned long j = 0; j < n; j++ )
        {
            coef[j] -= eta * gradient_coef[j];
            gradient_coef[j] = 0.0;
        }
        if ( fit_intercept == 1 )
        {
            *intercept -= eta * gradient_intercept;
            gradient_intercept = 0.0;
        }
    }
}

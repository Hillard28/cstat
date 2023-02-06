#include <stdlib.h>
#include "utils.h"
/*
// Gradient descent
void dgd( size_t m, size_t n, const double *X, const double *y, double *coef, double *intercept, double eta, int max_iter, int fit_intercept )
{
    double alpha = 1.0, beta = 0.0, gradient_intercept = 0.0;
    double eta_m = eta / (double)m;

    double *resid = (double *) malloc (m * sizeof(double));
    double *gradient_coef = (double *) malloc (n * sizeof(double));

    double y_hat;

    for ( size_t j = 0; j < n; j++ )
    {
        coef[j] = 0.0;
        gradient_coef[j] = 0.0;
    }
    *intercept = 0.0;
    
    for ( int epoch = 0; epoch < max_iter; epoch++ )
    {
        // Compute y_hat and gradients
        for ( size_t i = 0; i < m; i++ )
        {
            y_hat = dsigmoid( n, alpha, &X[n*i], coef, beta, *intercept );
            resid[i] = y[i] - y_hat;
            for ( size_t j = 0; j < n; j++ )
            {
                gradient_coef[j] -= (X[n*i + j] * resid[i]);
            }
            if ( fit_intercept == 1 )
            {
                gradient_intercept -= resid[i];
            }
        }
        // Adjust weights
        for ( size_t j = 0; j < n; j++ )
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
*/

void dgd( size_t m, size_t n, const double *X, const double *y, double *coef, double *intercept, double eta, int max_iter, int fit_intercept )
{
    double gradient_intercept = 0.0;
    double eta_m = eta / (double)m;

    double *resid = (double *) malloc (m * sizeof(double));
    double *gradient_coef = (double *) malloc (n * sizeof(double));

    double y_hat;

    for ( size_t j = 0; j < n; j++ )
    {
        coef[j] = 0.0;
        gradient_coef[j] = 0.0;
    }
    *intercept = 0.0;
    
    for ( int epoch = 0; epoch < max_iter; epoch++ )
    {
        if ( fit_intercept == 1 )
        {
            for ( size_t i = 0; i < m; i++ )
            {
                y_hat = dsigmoid( n, &X[n*i], coef, *intercept );
                resid[i] = y[i] - y_hat;
                for ( size_t j = 0; j < n; j++ )
                {
                    gradient_coef[j] -= (X[n*i + j] * resid[i]);
                }
                gradient_intercept -= resid[i];
            }
            for ( size_t j = 0; j < n; j++ )
            {
                coef[j] -= eta_m * gradient_coef[j];
                gradient_coef[j] = 0.0;
            }
            *intercept -= eta_m * gradient_intercept;
            gradient_intercept = 0.0;
        }
        else
        {
            for ( size_t i = 0; i < m; i++ )
            {
                y_hat = dsigmoid( n, &X[n*i], coef, *intercept );
                resid[i] = y[i] - y_hat;
                for ( size_t j = 0; j < n; j++ )
                {
                    gradient_coef[j] -= (X[n*i + j] * resid[i]);
                }
            }
            for ( size_t j = 0; j < n; j++ )
            {
                coef[j] -= eta_m * gradient_coef[j];
                gradient_coef[j] = 0.0;
            }
        }
    }
    free(resid);
    free(gradient_coef);
}

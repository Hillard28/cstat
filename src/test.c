#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "sgd.h"
#include "gd.h"
#include "utils.h"

int main(int argc, char **argv)
{
    double *X, *y, *coef, *y_preds;
    double intercept, eta;
    //double alpha = 1.0, beta = 1.0;
    unsigned long m = 100000;
    unsigned long n = 200;
    int max_iter;

    int class_0 = (unsigned long)(3.0 / 4.0 * (double)m);
    double pct_class_1 = 0.0;
    int n_odd = n % 2;

    if ( *argv[1] == '1' )
    {
        eta = 0.05;
        max_iter = 150;
    }
    else
    {
        eta = 0.5;
        max_iter = 250;
    }

    clock_t test_start;
    clock_t test_end;
    double test_time;

    printf("Constructing variables...\n");
    X = (double *) malloc (m * n * sizeof(double));
    y = (double *) malloc (m * sizeof(double));
    y_preds = (double *) malloc (m * sizeof(double));
    coef = (double *) malloc (n * sizeof(double));

    // Initialize classes
    for ( unsigned long i = 0; i < m; i++ )
    {
        if (i < class_0)
        {
            y[i] = 0.0;
        }
        else
        {
            y[i] = 1.0;
        }
        /*
        // Troubleshooting print
        if (i < 10 || i > m - 10)
        {
            //printf("%f\n", y[i]);
        }
        */
    }

    // Initialize observation features
    for ( unsigned long i = 0; i < m; i++ )
    {
        if (i < class_0)
        {
            X[n*i] = 1.0 / (double)m;
        }
        else
        {
            X[n*i] = (double)i / (double)m;
        }
        X[n*i + 1] = (double)i / (double)m;
        for ( unsigned long j = 2; j < n; j++ )
        {
            X[n*i + j] = (double)rand() / (double)RAND_MAX;
        }
        /*
        // Troubleshooting print
        if (i < 10)
        {
            //printf("%f\t%f\n", X[n*i], X[n*i+1]);
        }
        */
    }

    for ( unsigned long j = 0; j < n; j++ )
    {
        coef[j] = 0.0;
    }
    intercept = 0.0;

    // Fit weights
    test_start = clock();
    if ( *argv[1] == '1' )
    {
        printf("Running SGD...\n");
        dsgd( m, n, X, y, coef, &intercept, eta, max_iter, 1, -1 );
    }
    else
    {
        printf("Running GD...\n");
        dgd( m, n, X, y, coef, &intercept, eta, max_iter, 1 );
    }
    test_end = clock();
    test_time = (double)(test_end - test_start) / CLOCKS_PER_SEC;
    printf("Time taken: %f\n", test_time);

    // Compute y_hat and share of observations predicted as class 1
    printf("Making predictions...\n");
    for ( unsigned long i = 0; i < m; i++ )
    {
        y_preds[i] = dsigmoid( n, &X[i*n], coef, intercept, n_odd );
        //y_preds[i] = dsigmoid( n, alpha, &X[n*i], coef, beta, intercept );
    }

    printf("Printing results...\n");
    for ( unsigned long i = 0; i < m; i++ )
    {
        //printf("%f\n", y_pred[i]);
        if (y_preds[i] > 0.5)
        {
            pct_class_1 += 1.0;
        }
        
        // Troubleshooting print
        if (i < 10 || i > m - 10)
        {
            printf("%g\n", y_preds[i]);
        }
        
    }
    /*
    // Troubleshooting print
    printf("Intercept: %f\n", intercept);
    printf("Coefficients:\n");
    for ( unsigned long i = 0; i < n; i++ )
    {
        printf("%f\n", coef[i]);
    }
    */
    printf("Total observations: %ld\n", m);
    printf("Percent class 1: %f\n", pct_class_1 / (double)m);

    return 0;
}

// File gd.h
#ifndef GD_H
#define GD_H

// Stochastic gradient descent
void dgd( unsigned long m, unsigned long n, double *X, double *y, double *coef, double *intercept, double eta, int max_iter, int fit_intercept );

#endif

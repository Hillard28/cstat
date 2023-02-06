// File gd.h
#ifndef GD_H
#define GD_H

// Stochastic gradient descent
void dgd( size_t m, size_t n, const double *X, const double *y, double *coef, double *intercept, double eta, int max_iter, int fit_intercept );

#endif

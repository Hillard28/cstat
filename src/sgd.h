// File sgd.h
#ifndef SGD_H
#define SGD_H

// Stochastic gradient descent
void dsgd( unsigned long m, unsigned long n, double *X, double *y, double *coef, double *intercept, double eta, int max_iter, int fit_intercept, int random_seed );

#endif

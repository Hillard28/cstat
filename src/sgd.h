// File sgd.h
#ifndef SGD_H
#define SGD_H

// Stochastic gradient descent
void dsgd( size_t m, size_t n, const double *X, const double *y, double *coef, double *intercept, double eta, int max_iter, int fit_intercept, int random_seed );

#endif

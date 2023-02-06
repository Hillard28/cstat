# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 11:54:51 2023

@author: rgilland
"""

cdef extern from "sgd.h":
    double dsgd(size_t m, size_t n, const double *X, const double *y, double *coef, double *intercept, double eta, int max_iter, int fit_intercept, int random_seed)

cdef extern from "gd.h":
    double dgd(size_t m, size_t n, const double *X, const double *y, double *coef, double *intercept, double eta, int max_iter, int fit_intercept)

def sgd(m, n, X, y, coef, intercept, eta, max_iter, fit_intercept, random_seed):
    cdef const double[:] X_view = X
    cdef const double[:] y_view = y
    cdef double[:] coef_view = coef
    cdef double[:] intercept_view = intercept
    dsgd(m, n, &X_view[0], &y_view[0], &coef_view[0], &intercept_view[0], eta, max_iter, int(fit_intercept), random_seed)
    return coef, intercept

def gd(m, n, X, y, coef, intercept, eta, max_iter, fit_intercept):
    cdef const double[:] X_view = X
    cdef const double[:] y_view = y
    cdef double[:] coef_view = coef
    cdef double[:] intercept_view = intercept
    dgd(m, n, &X_view[0], &y_view[0], &coef_view[0], &intercept_view[0], eta, max_iter, int(fit_intercept))
    return coef, intercept

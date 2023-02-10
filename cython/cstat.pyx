# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 11:54:51 2023

@author: rgilland
"""

cdef extern from "sgd.h":
    void dsgd(size_t m, size_t n, const double *X, const double *y, double *coef, double *intercept, double eta, int max_iter, int fit_intercept, int random_seed)

cdef extern from "gd.h":
    void dgd(size_t m, size_t n, const double *X, const double *y, double *coef, double *intercept, double eta, int max_iter, int fit_intercept)

def sgd(size_t m, size_t n, const double[:] X, const double[:] y, double[:] coef, double[:] intercept, double eta, int max_iter, int fit_intercept, int random_seed):
    dsgd(m, n, &X[0], &y[0], &coef[0], &intercept[0], eta, max_iter, fit_intercept, random_seed)

def gd(size_t m, size_t n, const double[:] X, const double[:] y, double[:] coef, double[:] intercept, double eta, int max_iter, int fit_intercept):
    dgd(m, n, &X[0], &y[0], &coef[0], &intercept[0], eta, max_iter, fit_intercept)

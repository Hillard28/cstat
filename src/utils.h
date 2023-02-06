// File utils.h
#ifndef UTILS_H
#define UTILS_H

#define STATE_VECTOR_LENGTH 624
#define STATE_VECTOR_M      397

typedef struct tagMTRand {
  unsigned long mt[STATE_VECTOR_LENGTH];
  int index;
} MTRand;

MTRand seedRand(unsigned long seed);

unsigned long genRandLong(MTRand* rand);

double genRand(MTRand* rand);

double dlc( size_t n, const double *X, const double *coef, double intercept );

double dlcx2( size_t n, const double *X, const double *coef, double intercept );

double dlcx4( size_t n, const double *X, const double *coef, double intercept );

//double dsigmoid( size_t n, double alpha, const double *X, double *coef, double beta, double intercept );

double dsigmoid( size_t n, const double *X, double *coef, double intercept );

#endif

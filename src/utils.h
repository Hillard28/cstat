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

double dlc( unsigned long n, double *X, double *coef, double intercept );

double dsigmoid( unsigned long n, double alpha, double *X, double *coef, double beta, double intercept );

#endif

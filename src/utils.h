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

double dlcx2( size_t n, const double *X, const double *coef, double intercept, int n_odd );

//double dsigmoid( size_t n, double alpha, const double *X, double *coef, double beta, double intercept );

double dsigmoid( size_t n, const double *X, double *coef, double intercept, int n_odd );

// Function to swap two elements in the array
void swap(int* a, int* b);

// Function to partition the array and return the pivot index
int partition(double *feature_values, int low, int high);

// Quicksort function
void quicksort(double *feature_values, int low, int high);

// Function to drop duplicates
unsigned long drop_duplicates(double *feature_values, unsigned long m);

#endif

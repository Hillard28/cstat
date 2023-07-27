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

// Compute z = w * x + b
double dlc( unsigned long n, double *X, double *coef, double intercept );

// Compute y_hat = 1 / (1 + e^(-z))
double dsigmoid( unsigned long n, double alpha, double *X, double *coef, double beta, double intercept );

// Function to swap two elements in the array
void swap(int* a, int* b);

// Function to partition the array and return the pivot index
int partition(double *feature_values, int low, int high);

// Quicksort function
void quicksort(double *feature_values, int low, int high);

// Function to drop duplicates
unsigned long drop_duplicates(double *feature_values, unsigned long m);

#endif

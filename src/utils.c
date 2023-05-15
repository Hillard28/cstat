#include "utils.h"

// Mersenne Twister parameters
#define UPPER_MASK		0x80000000
#define LOWER_MASK		0x7fffffff
#define TEMPERING_MASK_B	0x9d2c5680
#define TEMPERING_MASK_C	0xefc60000

inline static void m_seedRand(MTRand* rand, unsigned long seed) {
  /* set initial seeds to mt[STATE_VECTOR_LENGTH] using the generator
   * from Line 25 of Table 1 in: Donald Knuth, "The Art of Computer
   * Programming," Vol. 2 (2nd Ed.) pp.102.
   */
  rand->mt[0] = seed & 0xffffffff;
  for(rand->index=1; rand->index<STATE_VECTOR_LENGTH; rand->index++) {
    rand->mt[rand->index] = (6069 * rand->mt[rand->index-1]) & 0xffffffff;
  }
}

/**
* Creates a new random number generator from a given seed.
*/
MTRand seedRand(unsigned long seed) {
  MTRand rand;
  m_seedRand(&rand, seed);
  return rand;
}

/**
 * Generates a pseudo-randomly generated long.
 */
unsigned long genRandLong(MTRand* rand) {

  unsigned long y;
  static unsigned long mag[2] = {0x0, 0x9908b0df}; /* mag[x] = x * 0x9908b0df for x = 0,1 */
  if(rand->index >= STATE_VECTOR_LENGTH || rand->index < 0) {
    /* generate STATE_VECTOR_LENGTH words at a time */
    int kk;
    if(rand->index >= STATE_VECTOR_LENGTH+1 || rand->index < 0) {
      m_seedRand(rand, 4357);
    }
    for(kk=0; kk<STATE_VECTOR_LENGTH-STATE_VECTOR_M; kk++) {
      y = (rand->mt[kk] & UPPER_MASK) | (rand->mt[kk+1] & LOWER_MASK);
      rand->mt[kk] = rand->mt[kk+STATE_VECTOR_M] ^ (y >> 1) ^ mag[y & 0x1];
    }
    for(; kk<STATE_VECTOR_LENGTH-1; kk++) {
      y = (rand->mt[kk] & UPPER_MASK) | (rand->mt[kk+1] & LOWER_MASK);
      rand->mt[kk] = rand->mt[kk+(STATE_VECTOR_M-STATE_VECTOR_LENGTH)] ^ (y >> 1) ^ mag[y & 0x1];
    }
    y = (rand->mt[STATE_VECTOR_LENGTH-1] & UPPER_MASK) | (rand->mt[0] & LOWER_MASK);
    rand->mt[STATE_VECTOR_LENGTH-1] = rand->mt[STATE_VECTOR_M-1] ^ (y >> 1) ^ mag[y & 0x1];
    rand->index = 0;
  }
  y = rand->mt[rand->index++];
  y ^= (y >> 11);
  y ^= (y << 7) & TEMPERING_MASK_B;
  y ^= (y << 15) & TEMPERING_MASK_C;
  y ^= (y >> 18);
  return y;
}

/**
 * Generates a pseudo-randomly generated double in the range [0..1].
 */
double genRand(MTRand* rand) {
  return((double)genRandLong(rand) / (unsigned long)0xffffffff);
}

// Compute z = w * x + b
double dlc( unsigned long n, double *X, double *coef, double intercept )
{
    double y_pred = intercept;
    for ( unsigned long i = 0; i < n; i++ )
    {
        y_pred += X[i] * coef[i];
    }
    return y_pred;
}

// Compute y_hat = 1 / (1 + e^(-z))
double dsigmoid( unsigned long n, double alpha, double *X, double *coef, double beta, double intercept )
{
    //double y_pred = intercept;
    //bli_ddotxv( BLIS_NO_CONJUGATE, BLIS_NO_CONJUGATE, n, &alpha, X, 1, coef, 1, &beta, &y_pred );
    double y_pred;
    y_pred = 1.0 / (1.0 + exp(-dlc(n, X, coef, intercept)));

    return y_pred;
}

// Function to swap two elements in the array
void swap(int* a, int* b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

// Function to partition the array and return the pivot index
int partition(double *feature_values, int low, int high) {
    int pivot = feature_values[high]; // Choosing the last element as the pivot
    int i = low - 1; // Index of the smaller element

    for (int j = low; j <= high - 1; j++) {
        // If the current element is smaller than or equal to the pivot
        if (feature_values[j] <= pivot) {
            i++;
            swap(&feature_values[i], &feature_values[j]);
        }
    }
    swap(&feature_values[i + 1], &feature_values[high]);
    return i + 1;
}

// Quicksort function
void quicksort(double *feature_values, int low, int high) {
    if (low < high) {
        // Partition the array and get the pivot index
        int pivotIndex = partition(feature_values, low, high);

        // Recursively sort the sub-arrays before and after the pivot
        quicksort(feature_values, low, pivotIndex - 1);
        quicksort(feature_values, pivotIndex + 1, high);
    }
}

// Function to drop duplicates
unsigned long drop_duplicates(double *feature_values, unsigned long m) {
    int i, j, k;
    for (i = 0; i < m; i++) {
        for (j = i + 1; j < m;) {
            if (feature_values[j] == feature_values[i]) {
                for (k = j; k < m - 1; k++) {
                    feature_values[k] = feature_values[k + 1];
                }
                m--;
            } else {
                j++;
            }
        }
    }
    return m;
}

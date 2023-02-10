#include <math.h>
#include "utils.h"
//#include "blis.h"

// Mersenne Twister parameters
#define UPPER_MASK		0x80000000
#define LOWER_MASK		0x7fffffff
#define TEMPERING_MASK_B	0x9d2c5680
#define TEMPERING_MASK_C	0xefc60000

// set initial seeds to mt[STATE_VECTOR_LENGTH]
inline static void m_seedRand(MTRand* rand, unsigned long seed) {
  rand->mt[0] = seed & 0xffffffff;
  for(rand->index=1; rand->index<STATE_VECTOR_LENGTH; rand->index++) {
    rand->mt[rand->index] = (6069 * rand->mt[rand->index-1]) & 0xffffffff;
  }
}

// Creates a new random number generator from a given seed.
MTRand seedRand(unsigned long seed) {
  MTRand rand;
  m_seedRand(&rand, seed);
  return rand;
}

// Generates a pseudo-randomly generated long.
unsigned long genRandLong(MTRand* rand) {

  unsigned long y;
  // mag[x] = x * 0x9908b0df for x = 0,1
  static unsigned long mag[2] = {0x0, 0x9908b0df};
  if(rand->index >= STATE_VECTOR_LENGTH || rand->index < 0) {
    // generate STATE_VECTOR_LENGTH words at a time
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

// Generates a pseudo-randomly generated double in the range [0..1].
double genRand(MTRand* rand) {
  return((double)genRandLong(rand) / (unsigned long)0xffffffff);
}

double dlcx2( size_t n, const double *X, const double *coef, double intercept, int n_odd )
{
  double res[2] = {intercept, 0.0};
  if (n_odd)
  {
    for ( size_t j = (n >> 1); j > 0; j-- )
    {
      res[0] += X[0] * coef[0];
      res[1] += X[1] * coef[1];
      X += 2;
      coef += 2;
    }
    return res[0] + res[1] + (X[0] * coef[0]);
  }
  else
  {
    for ( size_t j = (n >> 1); j > 0; j-- )
    {
      res[0] += X[0] * coef[0];
      res[1] += X[1] * coef[1];
      X += 2;
      coef += 2;
    }
    return res[0] + res[1];
  }
}

// Compute z = w * x + b
double dlc( size_t n, const double *X, const double *coef, double intercept )
{
    double z = intercept;
    for ( size_t j = 0; j < n; j++ )
    {
        z += X[j] * coef[j];
    }
    return z;
}

// Compute y_hat = 1 / (1 + e^(-(w * x + b)))
double dsigmoid( size_t n, const double *X, double *coef, double intercept, int n_odd )
{
    double z = dlcx2( n, X, coef, intercept, n_odd );
    
    if ( z >= 0)
    {
      return 1.0 / (1.0 + exp(-z));
    }
    else
    {
      z = exp(z);
      return z / (1.0 + z);
    }
}
/*
// Compute y_hat = 1 / (1 + e^(-(w * x + b)))
double dsigmoid( size_t n, double alpha, const double *X, double *coef, double beta, double intercept )
{
    double z = intercept;
    bli_ddotxv( BLIS_NO_CONJUGATE, BLIS_NO_CONJUGATE, n, &alpha, X, 1, coef, 1, &beta, &z );
    
    if ( z >= 0)
    {
      return 1.0 / (1.0 + exp(-z));
    }
    else
    {
      z = exp(z);
      return z / (1.0 + z);
    }
}
*/

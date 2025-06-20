# cstat
Basic statistical models in C

If X and y variables are NumPy arrays, you need to apply `numpy.ravel()` with `order='C'` to both, prior to computing coefficients and intercept. This is because matrices are treated as 1-dimensional arrays and indexed based on row and column stride. C code assumes row-major (C standard) order, which should be the NumPy default as well.

What is currently implemented is a standard logistic regression optimized using full or stochastic gradient descent. The stochastic gradient descent algorithm currently only supports a batch-size of 1.

Planned: optimization of full gradient descent (currently loses out to a NumPy-native, vectorized implementation), mini-batches for stochastic gradient descent, and implementation of the SAG/SAGA algorithms for gradient descent.

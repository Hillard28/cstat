# cstat
Statistical models in C

If X and y variables are NumPy arrays, you need to apply `numpy.ravel()` with `order='C'` to both, prior to computing coefficients and intercept. This is because matrices are treated as 1-dimensional arrays and indexed based on row and column stride. C code assumes row-major (C standard) order, which should be the NumPy default as well.

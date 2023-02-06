from setuptools import setup, Extension
from Cython.Build import cythonize

name="cstat"
sources=["cstat.pyx", "gd.c", "sgd.c", "utils.c"]
extensions = Extension(name, sources, extra_compile_args=["-O3", "-march=native"])

setup(ext_modules = cythonize(
     extensions,
     language_level=3,
))

from distutils.core import setup
from Cython.Build import cythonize

setup(name='ctomography', ext_modules=cythonize('tomography.pyx'))

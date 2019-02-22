from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
        "Nbody",
        ["Nbody.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],

    )
]

setup(name="Nbody",
      ext_modules=cythonize(ext_modules),
      include_dirs=[numpy.get_include()]
      )

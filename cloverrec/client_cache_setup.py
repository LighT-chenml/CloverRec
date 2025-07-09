from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        name="client_cache",       
        sources=["client_cache.cpp"], 
        extra_compile_args=["-std=c++11","-O2","-w"],
    ),
]

setup(
    name="client_cache",
    version="1.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
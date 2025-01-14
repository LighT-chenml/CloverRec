from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        name="pim_module",       
        sources=["pim_module.cpp"], 
        extra_compile_args=["-std=c++11","-w","-I/usr/include/dpu"],
        extra_link_args=["-ldpu"],
    ),
]

setup(
    name="pim_module",
    version="1.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
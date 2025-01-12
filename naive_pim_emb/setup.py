from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "pim_module",        # Python 模块名称
        ["pim_module.cpp"],  # C++ 源文件
    ),
]

setup(
    name="pim_module",
    version="1.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
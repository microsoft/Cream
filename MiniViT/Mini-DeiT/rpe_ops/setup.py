"""Build iRPE (image RPE) Functions"""
from setuptools import setup, Extension
import torch
from torch.utils import cpp_extension

ext_t = cpp_extension.CppExtension
ext_fnames = ['rpe_index.cpp']
define_macros = []
extra_compile_args = dict(cxx=['-fopenmp', '-O3'],
                          nvcc=['-O3'])

if torch.cuda.is_available():
    ext_t = cpp_extension.CUDAExtension
    ext_fnames.append('rpe_index_cuda.cu')
    define_macros.append(('WITH_CUDA', None))

setup(name='rpe_index',
      version="1.2.0",
      ext_modules=[ext_t(
                   'rpe_index_cpp',
                   ext_fnames,
                   define_macros=define_macros,
                   extra_compile_args=extra_compile_args,
                   )],
      cmdclass={'build_ext': cpp_extension.BuildExtension})

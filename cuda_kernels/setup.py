"""
GraphSnapShot CUDA Kernels Setup
Python bindings for GraphSnapShot CUDA kernels
"""

from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11
import os
import torch

# Get CUDA paths
def find_cuda():
    """Find CUDA installation"""
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home is None:
        raise RuntimeError("CUDA_HOME or CUDA_PATH environment variable not set")
    
    cuda_include = os.path.join(cuda_home, 'include')
    cuda_lib = os.path.join(cuda_home, 'lib64')
    
    if not os.path.exists(cuda_include):
        raise RuntimeError(f"CUDA include directory not found: {cuda_include}")
    
    return cuda_home, cuda_include, cuda_lib

# Get torch CUDA paths
def get_torch_cuda_paths():
    """Get CUDA paths from PyTorch"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available in PyTorch")
    
    torch_cuda_home = torch.utils.cpp_extension.CUDA_HOME
    torch_include = torch.utils.cpp_extension.include_paths()
    torch_libs = torch.utils.cpp_extension.library_paths()
    
    return torch_cuda_home, torch_include, torch_libs

# Find CUDA and Torch paths
try:
    cuda_home, cuda_include, cuda_lib = find_cuda()
    torch_cuda_home, torch_include, torch_libs = get_torch_cuda_paths()
except RuntimeError as e:
    print(f"Warning: {e}")
    print("Falling back to CPU-only compilation")
    cuda_home = cuda_include = cuda_lib = None
    torch_cuda_home = torch_include = torch_libs = None

# Define extensions
ext_modules = []

# CUDA extension
if cuda_home is not None:
    graphsnapshot_cuda = Pybind11Extension(
        'graphsnapshot_cuda',
        sources=[
            'graphsnapshot_kernels.cu',
            'python_bindings.cpp'
        ],
        include_dirs=[
            pybind11.get_cmake_dir() + "/../../../include",
            cuda_include,
            torch_include,
            '..',
        ],
        libraries=['cudart', 'curand', 'cub'],
        library_dirs=[cuda_lib, torch_libs],
        language='c++',
        cxx_std=14,
        extra_compile_args=[
            '-DWITH_CUDA',
            '-DUSE_CUDA',
            '--expt-relaxed-constexpr',
            '--extended-lambda',
            '-O3',
            '-DNDEBUG'
        ],
        extra_link_args=[
            '-lcudart',
            '-lcurand',
            '-lcub'
        ]
    )
    ext_modules.append(graphsnapshot_cuda)

# CPU fallback extension
graphsnapshot_cpu = Pybind11Extension(
    'graphsnapshot_cpu',
    sources=[
        'cpu_fallback.cpp'
    ],
    include_dirs=[
        pybind11.get_cmake_dir() + "/../../../include",
        '..',
    ],
    language='c++',
    cxx_std=14,
    extra_compile_args=[
        '-O3',
        '-DNDEBUG'
    ]
)
ext_modules.append(graphsnapshot_cpu)

setup(
    name='graphsnapshot-kernels',
    version='1.0.0',
    description='GraphSnapShot CUDA Kernels',
    author='GraphSnapShot Team',
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires='>=3.7',
    install_requires=[
        'torch>=1.9.0',
        'pybind11>=2.6.0',
        'numpy>=1.19.0',
    ],
    extras_require={
        'cuda': [
            'cupy>=9.0.0',
        ]
    }
)

import os
import torch
import glob
from setuptools import find_packages, setup
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension, BuildExtension

library_name = "mzi_onn_sim"
name_sources_cpp = "*_core.cpp"
name_sources_cuda = "*_core_cuda.cu"


def get_extensions():
    use_cuda = os.getenv("USE_CUDA", "1") == "1"
    use_cuda = use_cuda and torch.cuda.is_available() and CUDA_HOME is not None
    extension = CUDAExtension if use_cuda else CppExtension

    this_dir = os.path.dirname(os.path.curdir)
    extensions_dir = os.path.join(this_dir, library_name)
    sources = list(glob.glob(os.path.join(extensions_dir, name_sources_cpp)))
    sources_cuda = list(glob.glob(os.path.join(extensions_dir, name_sources_cuda)))
    if use_cuda:
        sources += sources_cuda

    extra_compile_args = {
        "cxx": [
            "-O2",
            "-DPy_LIMITED_API=0x03090000",
        ],
        "nvcc": [
            "-O3",
        ],
    }
    ext_modules = [
        extension(
            f"{library_name}._C",
            sources,
            extra_compile_args=extra_compile_args,
            py_limited_api=True,
        )
    ]
    return ext_modules


setup(
    name=library_name,
    version="0.1.0",
    packages=find_packages(),
    ext_modules=get_extensions(),
    install_requires=["torch"],
    cmdclass={'build_ext': BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
)

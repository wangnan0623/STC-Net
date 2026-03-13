import os
os.environ['PATH'] = os.environ['PATH'] + ':/home/wangnan/.conda/envs/AHNet/bin'
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# 获取 PyTorch 的 CUDA 头文件路径
torch_cuda_include_dir = os.path.join(os.path.dirname(torch.__file__), 'include')

print("PATH:", os.environ['PATH'])


setup(
    name='_prroi_pooling',
    ext_modules=[
        CUDAExtension(
            name='_prroi_pooling',
            sources=[
                os.path.abspath('src/prroi_pooling_gpu.c'),
                os.path.abspath('src/prroi_pooling_gpu_impl.cu'),
            ],
            include_dirs=[
                os.path.abspath('src'),
                torch_cuda_include_dir,  # PyTorch 的头文件路径
                # os.path.join(torch_cuda_include_dir, 'THC'),  # THC 的头文件路径（已移除）
            ],
            extra_compile_args={
                'cxx': ['-g', '-O2'],
                'nvcc': ['-g', '-O2', '-arch=sm_75']
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
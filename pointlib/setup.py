from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="pointlib",
    ext_modules=[
        CUDAExtension(
            "pointlib",
            [
            'src/bindings.cpp',
            'src/knn_points.cpp',
            'src/knn_points_gpu.cu',
            'src/sampling.cpp',
            'src/sampling_gpu.cu',
            'src/group_points.cpp',
            'src/group_points_gpu.cu',
            ],
            extra_compile_args={
                "cxx": ["-g"],
                "nvcc": ["-O2"]})
    ],
    cmdclass={"build_ext": BuildExtension},
)

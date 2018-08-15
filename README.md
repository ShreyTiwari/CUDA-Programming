# CUDA-Programming

This repository contains a few examples of programs that can be accelerated with the help of GPU's.
The programs must be complied using the Nvidia CUDA compiler( nvcc ). The programs can be easily profiled with the help of
nvidia's built-in profilers such as the nvprof or nvidia visual profiler.

Note that the programs make use of unified memory concept and hence you will need a nvidia graphics card that supports this,
i.e can run CUDA versions 6 or above.

It is better to manually manage the memory and data transfer, but for learning purposes( and due the simplicity of the programs )
the unified memory is used.

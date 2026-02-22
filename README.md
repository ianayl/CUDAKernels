# CUDA Kernel Experiments

Messing around with different GPU kernel impls in CUDA to see how many FLOPS I can get out of my handwritten kernels

I am currently too ~~lazy~~ preoccupied with coursework to write a makefile or cmake:
```sh
nvcc -arch=compute_120 src/test.cu -Iinclude src/Timer.cpp src/Kernels/*
```

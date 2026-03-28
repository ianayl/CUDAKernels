# Internals

- `Adhoc` contain ad-hoc kernels: Great for scrunching something up quickly, but not great for anything else: i.e. end-user experience or managing large-scale projects.
- `Kernels` contain kernels written with abstractions, hopefully able to accommodate modular warp/block/device/etc. logic in the future, as is done in production code.

I am currently ~~lazy~~ preoccupied with coursework, so haven't written a makefile:
```sh
nvcc -arch=compute_120 src/main.cu -Iinclude src/Timer.cu src/Adhoc/* src/Kernels/GEMM/* src/Kernels/Impl/*
```
Don't forget to change `-arch` and whatnot.

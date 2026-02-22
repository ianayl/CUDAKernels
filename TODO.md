# TODO

- write a jig that benchmarks each GEMM kernel
  - write a function that generates random matrices
  - write a jig that:
    - takes each kernel and inputs different sized matrices into the kernels
      - kernels need standardized function signature
    - measures time
    - figures out GFLOPS for computation
  - write another jig that
    - takes average GFLOPS and time and puts it on a python graph 

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
- write testing for your GPU kernels

# Performance considerations

- kernel launch parameters could muddy performance results
  - kernels should be autotuned every execution to ensure kernel launch parameters is not an issue
  - Q: How does autotuning compare to taking cudaOccupancyMaxPotentialBlockSize and maximizing?

- isolate off OS multipurpose cores (2nd NUMA node?)
- do warmup kernels before measuring (ramp GPU clock speeds, prevent cold cache misses, yadiyadiyada.) 
- run multiple times and grab median (duh)
- option to measure based off of cpu cycles or wall time (or can I fetch GPU cycles?)
- option to measure empty kernel submission, then "correct" measurements?
  - Q: is this really necessary if it is a constant cost across all measurements?

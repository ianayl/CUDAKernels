#pragma once

#ifndef BENCHMARKS_GEMM_HPP
#define BENCHMARKS_GEMM_HPP

#include <Benchmarks/Impl/Bench.hpp>

// READ BEFORE FURTHER DEVELOPMENT
// - Each test should be its own executable
// - write out one of the executables first and see what you actually want
// - ... *then* see what needs to go into a wrapper class so you don't write
//   6 million kernel launches everytime

namespace Benchmarks {
  class GEMM : public Impl::Bench {
  public:
  private:
  };
}

// 

#endif // BENCHMARKS_GEMM_HPP

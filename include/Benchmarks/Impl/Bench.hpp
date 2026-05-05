#pragma once

#ifndef BENCHMARKS_IMPL_BENCH_HPP
#define BENCHMARKS_IMPL_BENCH_HPP

#include <Benchmarks/Result.hpp>

namespace Benchmarks::Impl {
  class Bench {
    using Result = Benchmarks::Result;
  public:
    std::string getCSV();

    std::string testName;
    // Do I want inputSizeType?
    // Do I want to control what data to output?
  private:
    void pushResult();
    std::vector<Result> results;
  };
}

#endif // BENCHMARKS_IMPL_BENCH_HPP

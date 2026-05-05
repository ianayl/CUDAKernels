#pragma once

#ifndef BENCHMARKS_RESULT_HPP
#define BENCHMARKS_RESULT_HPP

#include <string>

// Potentially useful fields:
// - [ ] test name -- should be in test container
// - [ ] kernel name
// - [ ] input size
// - [ ] input size type -- should be in test container
// - [ ] is baseline? -- should be handled by test container
// What measurement unit to output in the final CSV shoudl probably be decided
// by the test container -- for now, include all the types
// - [ ] time elapsed
// - [ ] time elapsed unit
// - [ ] cpu cycles
// - [ ] gpu cycles
// - [ ] FLOPS

namespace Benchmarks {
  struct Result {
    std::string kernelName;
    long inputSize;
    long time;
    std::string timeUnit;
    long flops;
    long cpuCycles;
    long gpuCycles;
  };
}

#endif // BENCHMARKS_RESULT_HPP

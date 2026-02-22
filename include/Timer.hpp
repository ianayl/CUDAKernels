#pragma once

#ifndef TIMER_HPP
#define TIMER_HPP

#include <chrono>

class Timer {
  using Clock = std::chrono::high_resolution_clock;

public:
  Timer(); // Also starts the timer
  Clock::duration stop();
  Clock::duration get();
  int getNs();

private:
  Clock::time_point start;
  Clock::time_point end;
};

#endif // TIMER_HPP

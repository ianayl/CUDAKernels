#pragma once

#ifndef CONFIG_HPP
#define CONFIG_HPP

struct Config {
	bool timer = true;
};

inline constexpr Config DefaultKernelConfig{};

#endif // CONFIG_HPP

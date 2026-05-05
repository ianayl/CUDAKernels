#include <Tensor.cuh>

#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <algorithm>

template<typename T>
void Tensor<T>::push() {
  if (nullptr == d_vec) cudaMalloc(&d_vec, sizeof(T)*size);
  cudaMemcpy(d_vec, h_vec.data(), sizeof(T)*size, cudaMemcpyHostToDevice);
  // TODO: check errors
}

template<typename T>
void Tensor<T>::pull() {
  // TODO: should warn or assert
  if (nullptr == d_vec) return;
  cudaMemcpy(h_vec.data(), d_vec, sizeof(T)*size, cudaMemcpyDeviceToHost);
  // TODO: check errors
}

template<typename T>
Tensor<T>& Tensor<T>::operator=(Tensor<T>&& other) noexcept {
  if (this != &other) {
    size = other.size;
    h_vec = std::move(other.h_vec);
    d_vec = other.d_vec;
    dim_x = other.dim_x;
    dim_y = other.dim_y;
    dim_z = other.dim_z;

    // TODO: handle our existing h_vec and d_vec
    other.d_vec = nullptr;
    other.size = 0;
  }
  return *this;
}

template<typename T>
bool Tensor<T>::equals(const Tensor& other) {
  if (dim_x != other.dim_x || dim_y != other.dim_y || dim_z != other.dim_z)
    return false;

  for (size_t z = 0; z < dim_z; z++) {
    for (size_t y = 0; y < dim_y; y++) {
      for (size_t x = 0; x < dim_x; x++)
        if (h_vec[z*dim_y + y*dim_x + x] != other[z*dim_y + y*dim_x + x])
          return false;
    }
  }
  return true;
}

template<typename T>
void Tensor<T>::print() {
  for (size_t z = 0; z < dim_z; z++) {
    // TODO: Think about how to print z delimiters
    for (size_t y = 0; y < dim_y; y++) {
      for (size_t x = 0; x < dim_x; x++)
        std::cout << h_vec[z*dim_y + y*dim_x + x] << " ";
      std::cout << "\n";
    }
  }
}

template<typename T>
Tensor<T> Tensor<T>::random(size_t y, size_t x) {
  // TODO Do I want to make this multithreaded for large datasets?
  static std::mt19937 mt{ std::random_device{}() }; // use thread_local?
  static std::uniform_int_distribution dist{ 0, 20 };
  std::vector<T> data(y*x);
  std::generate(data.begin(), data.end(), [&]() { return dist(mt); });
  return Tensor<T>{y, x, std::move(data)};
}

template<typename T>
Tensor<T> Tensor<T>::random(size_t s) {
  return random(1, s);
}


// Explicit instantiations
template class Tensor<float>;

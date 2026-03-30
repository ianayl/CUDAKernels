#include <Tensor.cuh>

#include <cuda_runtime.h>
#include <iostream>

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

// Explicit instantiations
template class Tensor<float>;

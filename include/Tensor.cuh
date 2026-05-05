#pragma once

#ifndef TENSOR_CUH
#define TENSOR_CUH

#include <vector>

template<typename T>
class Tensor {
public:
  Tensor(size_t s)
    : h_vec(s), size(s), dim_x(s), dim_y(1), dim_z(1) { }
  // Tensor(size_t rows, size_t cols)
  // Tensor(size_t height, size_t width)
  Tensor(size_t y, size_t x)
    : h_vec(x * y), size(x * y), dim_x(x), dim_y(y), dim_z(1) { }

  Tensor(size_t s, const T& val)
    : h_vec(s, val), size(s), dim_x(s), dim_y(1), dim_z(1) { }
  Tensor(size_t y, size_t x, const T& val)
    : h_vec(x * y, val), size(x * y), dim_x(x), dim_y(y), dim_z(1) { }

  // TODO: these constructors could benefit from input vector size checks
  Tensor(size_t s, std::initializer_list<T> init)
    : h_vec(init), size(s), dim_x(s), dim_y(s), dim_z(1) { }
  Tensor(size_t y, size_t x, std::initializer_list<T> init)
    : h_vec(init), size(x * y), dim_x(x), dim_y(y), dim_z(1) { }

  Tensor(size_t s, const std::vector<T>& vec)
    : h_vec(vec), size(s), dim_x(s), dim_y(1), dim_z(1) { }
  Tensor(size_t y, size_t x, const std::vector<T>& vec)
    : h_vec(vec), size(x * y), dim_x(x), dim_y(y), dim_z(1) { }

  Tensor(size_t s, std::vector<T>&& vec)
    : h_vec(std::move(vec)), size(s), dim_x(s), dim_y(1), dim_z(1) { }
  Tensor(size_t y, size_t x, std::vector<T>&& vec)
    : h_vec(std::move(vec)), size(x * y), dim_x(x), dim_y(y), dim_z(1) { }

  T& operator[](size_t x) {
    return h_vec[x];
  }
  T operator[](size_t x) const {
    return h_vec[x];
  }
  T& operator()(size_t y, size_t x) {
    return h_vec[y * dim_x + x];
  }
  T operator()(size_t y, size_t x) const {
    return h_vec[y * dim_x + x];
  }

  T& at(size_t x) {
    // TODO: assert x < size
    return h_vec[x];
  }
  T& at(size_t y, size_t x) {
    // TODO: assert index < size
    return h_vec[y * dim_x + x];
  }

  T* operator*() { return d_vec; }
  T* data() { return h_vec.data(); }

  // TODO: function to check if d_vec is initialized?

  void push();
  void pull();
  bool equals(const Tensor& other);
  bool operator==(const Tensor& other) { return equals(other); }
  void print();

  // Copy constructors -- disable for now
  // TODO: copies should create entirely new memory buffers
  // implement if I ever need it...
  Tensor(const Tensor&) = delete;
  Tensor& operator=(const Tensor&) = delete;

  // Move constructors / assignments
  Tensor(Tensor&& other) noexcept
    : size(other.size), h_vec(std::move(other.h_vec)), d_vec(other.d_vec),
      dim_x(other.dim_x), dim_y(other.dim_y), dim_z(other.dim_z) { }
  Tensor& operator=(Tensor&& other) noexcept;

  static Tensor random(size_t s);
  static Tensor random(size_t y, size_t x);
  static Tensor random(size_t z, size_t y, size_t x);

  ~Tensor() {
    if (d_vec) cudaFree(d_vec);
  }

  size_t size;
  size_t dim_x;
  size_t dim_y;
  size_t dim_z;
private:
  std::vector<T> h_vec;
  T* d_vec = nullptr;
};

#endif // TENSOR_CUH

// TODO: TensorView class?
// - pushes a tensor upon construction
// - pulls upon destruction

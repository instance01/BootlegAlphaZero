#ifndef UTIL_HEADER
#define UTIL_HEADER
#include <torch/torch.h>


template<typename T>
std::vector<T> flatten(const std::vector<std::vector<T>> &orig)
{
  std::vector<T> ret;
  for(const auto &v: orig)
    ret.insert(ret.end(), v.begin(), v.end());
  return ret;
}

template<typename T>
std::vector<float> flatten_as_float(const std::vector<std::vector<T>> &orig)
{
  std::vector<float> ret;
  for(const auto &v: orig)
    ret.insert(ret.end(), v.begin(), v.end());
  return ret;
}

template<typename T>
torch::Tensor vec_1d_as_tensor(std::vector<T>& vec, c10::ScalarType dtype) {
  std::vector<int64_t> sizes_ = {1, static_cast<int64_t>(vec.size())};
  at::IntArrayRef sizes = at::IntArrayRef(sizes_);
  auto tensor = torch::from_blob(
      vec.data(),
      sizes,
      torch::TensorOptions().dtype(dtype)
  );
  return tensor;
}

template<typename T>
torch::Tensor vec_2d_as_tensor(std::vector<T>& vec, c10::ScalarType dtype, size_t size1, size_t size2) {
  std::vector<int64_t> sizes_ = {
    static_cast<int64_t>(size1), static_cast<int64_t>(size2)
  };
  at::IntArrayRef sizes = at::IntArrayRef(sizes_);
  auto tensor = torch::from_blob(vec.data(), sizes, torch::TensorOptions().dtype(dtype));
  return tensor;
}
#endif

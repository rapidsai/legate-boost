/* Copyright 2023 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */
#pragma once
#include "legate_library.h"
#include "legateboost.h"
#ifdef __CUDACC__
#include <thrust/binary_search.h>
#endif
#include <cstddef>
#include <utility>
#include <tuple>
#include <tcb/span.hpp>

namespace legateboost {

inline const double eps = 1e-5;  // Add this term to the hessian to prevent division by zero

template <typename T>
struct GPairBase {
  using value_type = T;
  T grad           = 0.0;
  T hess           = 0.0;

  __host__ __device__ auto operator+=(const GPairBase<T>& b) -> GPairBase<T>&
  {
    this->grad += b.grad;
    this->hess += b.hess;
    return *this;
  }
};

template <typename T>
inline __host__ __device__ auto operator-(const GPairBase<T>& a, const GPairBase<T>& b)
  -> GPairBase<T>
{
  return GPairBase<T>{a.grad - b.grad, a.hess - b.hess};
}

template <typename T>
inline __host__ __device__ auto operator+(const GPairBase<T>& a, const GPairBase<T>& b)
  -> GPairBase<T>
{
  return GPairBase<T>{a.grad + b.grad, a.hess + b.hess};
}

using GPair        = GPairBase<double>;
using IntegerGPair = GPairBase<int64_t>;

// Some helpers for indexing into a binary tree
class BinaryTree {
 public:
  __host__ __device__ static auto Parent(int i) -> int { return (i - 1) / 2; }
  __host__ __device__ static auto LeftChild(int i) -> int { return (2 * i) + 1; }
  __host__ __device__ static auto RightChild(int i) -> int { return (2 * i) + 2; }
  __host__ __device__ static auto Sibling(int i) -> int { return (i % 2 == 0) ? i - 1 : i + 1; }
  __host__ __device__ static auto LevelBegin(int level) -> int { return (1 << level) - 1; }
  __host__ __device__ static auto LevelEnd(int level) -> int { return (1 << (level + 1)) - 1; }
  __host__ __device__ static auto NodesInLevel(int level) -> int { return 1 << level; }
};

// Estimate if the left or right child has less data
// We compute the histogram for the child with less data
// And infer the other side by subtraction from the parent
template <typename GPairT>
inline __host__ __device__ auto SelectHistogramNode(int parent,
                                                    const legate::Buffer<GPairT, 2>& node_sums)
  -> std::pair<int, int>
{
  int const left_child  = BinaryTree::LeftChild(parent);
  int const right_child = BinaryTree::RightChild(parent);
  if (node_sums[{left_child, 0}].hess < node_sums[{right_child, 0}].hess) {
    return {left_child, right_child};
  }
  return {right_child, left_child};
}

template <typename GPairT>
inline __host__ __device__ auto ComputeHistogramBin(int node_id,
                                                    const legate::Buffer<GPairT, 2>& node_sums,
                                                    bool parent_histogram_exists) -> bool
{
  if (node_id == 0) { return true; }
  if (node_id < 0) { return false; }
  if (!parent_histogram_exists) { return true; }

  int const parent                     = BinaryTree::Parent(node_id);
  auto [histogram_node, subtract_node] = SelectHistogramNode(parent, node_sums);
  return histogram_node == node_id;
}

__host__ __device__ inline auto CalculateLeafValue(double G, double H, double alpha) -> double
{
  return -G / (H + alpha);
}

// Container for the CSR matrix containing the split proposals
template <typename T>
class SparseSplitProposals {
 public:
  legate::Buffer<T, 1> split_proposals;
  legate::Buffer<int32_t, 1> row_pointers;
  int32_t num_features;
  int32_t histogram_size;
  SparseSplitProposals(const legate::Buffer<T, 1>& split_proposals,
                       const legate::Buffer<int32_t, 1>& row_pointers,
                       int32_t num_features,
                       int32_t histogram_size)
    : split_proposals(split_proposals),
      row_pointers(row_pointers),
      num_features(num_features),
      histogram_size(histogram_size)
  {
  }

// Returns the bin index for a given feature and value
// If the value is not in the split proposals, -1 is returned
#ifdef __CUDACC__
  __device__ auto FindBin(T x, int feature) const -> int
  {
    auto feature_row_begin = row_pointers[feature];
    auto feature_row_end   = row_pointers[feature + 1];
    auto* ptr              = thrust::lower_bound(
      thrust::seq, split_proposals.ptr(feature_row_begin), split_proposals.ptr(feature_row_end), x);
    return ptr - split_proposals.ptr(0);
  }
#else
  [[nodiscard]] auto FindBin(T x, int feature) const -> int
  {
    auto feature_row_begin = legate::coord_t{row_pointers[feature]};
    auto feature_row_end   = legate::coord_t{row_pointers[feature + 1]};
    auto* ptr              = std::lower_bound(
      split_proposals.ptr(feature_row_begin), split_proposals.ptr(feature_row_end), x);
    return ptr - split_proposals.ptr(0);
  }
#endif

#ifdef __CUDACC__
  __host__ __device__ auto FindFeature(int bin_idx) const -> int
  {
    // Binary search for the feature
    return thrust::upper_bound(
             thrust::seq, row_pointers.ptr(0), row_pointers.ptr(num_features), bin_idx) -
           row_pointers.ptr(0) - 1;
  }
#endif

  [[nodiscard]] __host__ __device__ auto FeatureRange(int feature) const -> std::tuple<int, int>
  {
    return std::make_tuple(row_pointers[feature], row_pointers[feature + 1]);
  }
};

template <typename GPairT>
class Histogram {
 public:
  using value_type = GPairT;
  // If we are using int64 as our type we need to do atomic adds as unsigned long long
  using atomic_add_type =
    std::conditional_t<std::is_same_v<typename value_type::value_type, int64_t>,
                       unsigned long long,  // NOLINT(runtime/int)
                       typename value_type::value_type>;

 private:
  legate::Buffer<GPairT, 3> buffer_;  // Nodes, outputs, bins
  int node_begin_{};
  int node_end_{};
  std::size_t size_{};

 public:
#ifdef __CUDACC__
  Histogram(int node_begin, int node_end, int num_outputs, int num_bins, cudaStream_t stream)
    : buffer_(legate::create_buffer<GPairT, 3>({node_end - node_begin, num_outputs, num_bins})),
      node_begin_(node_begin),
      node_end_(node_end),
      size_(static_cast<std::size_t>((node_end - node_begin) * num_outputs * num_bins))
  {
    static_assert(sizeof(GPairT) == 2 * sizeof(typename GPairT::value_type),
                  "Unexpected size of GPairT");
    static_assert(sizeof(GPairT) == 2 * sizeof(atomic_add_type),
                  "AtomicAdd type does not match size of type");

    CHECK_CUDA(
      cudaMemsetAsync(buffer_.ptr(legate::Point<3>::ZEROES()), 0, size_ * sizeof(GPairT), stream));
  }
#else
  Histogram(int node_begin, int node_end, int num_outputs, int num_bins)
    : buffer_(legate::create_buffer<GPairT, 3>({node_end - node_begin, num_outputs, num_bins})),
      node_begin_(node_begin),
      node_end_(node_end),
      size_(static_cast<std::size_t>((node_end - node_begin) * num_outputs * num_bins))
  {
    static_assert(sizeof(GPairT) == 2 * sizeof(typename GPairT::value_type),
                  "Unexpected size of GPairT");

    tcb::span<GPairT> span(buffer_.ptr(legate::Point<3>::ZEROES()), size_);
    for (auto& g : span) { g = GPairT{0.0, 0.0}; }
  }
#endif
  Histogram() = default;

  void Destroy()
  {
    if (size_ > 0) { buffer_.destroy(); }
    node_begin_ = 0;
    node_end_   = 0;
    size_       = 0;
  }

  auto ContainsBatch(int node_begin_idx, int node_end_idx) -> bool
  {
    return node_begin_idx >= node_begin_ && node_end_idx <= node_end_;
  }

  __device__ auto ContainsNode(int node_idx) -> bool
  {
    return node_idx >= node_begin_ && node_idx < node_end_;
  }

  auto Ptr(int node_idx) -> GPairT* { return buffer_.ptr({node_idx - node_begin_, 0, 0}); }

  auto Size() -> std::size_t { return size_; }

  // Node, output, bin
  __host__ __device__ auto operator[](legate::Point<3> p) -> GPairT&
  {
    return buffer_[{p[0] - node_begin_, p[1], p[2]}];
  }
};

class BuildTreeTask : public Task<BuildTreeTask, BUILD_TREE> {
 public:
  static void cpu_variant(legate::TaskContext context);
#ifdef LEGATEBOOST_USE_CUDA
  static void gpu_variant(legate::TaskContext context);
#endif
};

}  // namespace legateboost

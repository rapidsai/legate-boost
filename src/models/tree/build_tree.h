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

namespace legateboost {

inline const double eps = 1e-5;  // Add this term to the hessian to prevent division by zero

// Some helpers for indexing into a binary tree
class BinaryTree {
 public:
  __host__ __device__ static int Parent(int i) { return (i - 1) / 2; }
  __host__ __device__ static int LeftChild(int i) { return 2 * i + 1; }
  __host__ __device__ static int RightChild(int i) { return 2 * i + 2; }
  __host__ __device__ static int Sibling(int i) { return (i % 2 == 0) ? i - 1 : i + 1; }
  __host__ __device__ static int LevelBegin(int level) { return (1 << level) - 1; }
  __host__ __device__ static int LevelEnd(int level) { return (1 << (level + 1)) - 1; }
  __host__ __device__ static int NodesInLevel(int level) { return 1 << level; }
};

// Estimate if the left or right child has less data
// We compute the histogram for the child with less data
// And infer the other side by subtraction from the parent
inline __host__ __device__ std::pair<int, int> SelectHistogramNode(
  int parent, legate::Buffer<double, 2> node_hessians)
{
  int left_child  = BinaryTree::LeftChild(parent);
  int right_child = BinaryTree::RightChild(parent);
  if (node_hessians[{left_child, 0}] < node_hessians[{right_child, 0}]) {
    return {left_child, right_child};
  }
  return {right_child, left_child};
}

inline __host__ __device__ bool ComputeHistogramBin(int node_id,
                                                    legate::Buffer<double, 2> node_hessians,
                                                    bool parent_histogram_exists)
{
  if (node_id == 0) return true;
  if (node_id < 0) return false;
  if (!parent_histogram_exists) return true;

  int parent                           = BinaryTree::Parent(node_id);
  auto [histogram_node, subtract_node] = SelectHistogramNode(parent, node_hessians);
  return histogram_node == node_id;
}

__host__ __device__ inline double CalculateLeafValue(double G, double H, double alpha)
{
  return -G / (H + alpha);
}

template <typename T>
struct GPairBase {
  T grad = 0.0;
  T hess = 0.0;

  __host__ __device__ GPairBase<T>& operator+=(const GPairBase<T>& b)
  {
    this->grad += b.grad;
    this->hess += b.hess;
    return *this;
  }
};

template <typename T>
inline __host__ __device__ GPairBase<T> operator-(const GPairBase<T>& a, const GPairBase<T>& b)
{
  return GPairBase<T>{a.grad - b.grad, a.hess - b.hess};
}

template <typename T>
inline __host__ __device__ GPairBase<T> operator+(const GPairBase<T>& a, const GPairBase<T>& b)
{
  return GPairBase<T>{a.grad + b.grad, a.hess + b.hess};
}

using GPair        = GPairBase<double>;
using IntegerGPair = GPairBase<int64_t>;

// Container for the CSR matrix containing the split proposals
template <typename T>
class SparseSplitProposals {
 public:
  legate::Buffer<T, 1> split_proposals;
  legate::Buffer<int32_t, 1> row_pointers;
  int32_t num_features;
  int32_t histogram_size;
  static const int NOT_FOUND = -1;
  SparseSplitProposals(legate::Buffer<T, 1> split_proposals,
                       legate::Buffer<int32_t, 1> row_pointers,
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
  __device__ int FindBin(T x, int feature) const
  {
    auto feature_row_begin = row_pointers[feature];
    auto feature_row_end   = row_pointers[feature + 1];
    auto ptr               = thrust::lower_bound(
      thrust::seq, split_proposals.ptr(feature_row_begin), split_proposals.ptr(feature_row_end), x);
    if (ptr == split_proposals.ptr(feature_row_end)) return NOT_FOUND;
    return ptr - split_proposals.ptr(0);
  }
#else
  int FindBin(T x, int feature) const
  {
    auto feature_row_begin = row_pointers[feature];
    auto feature_row_end   = row_pointers[feature + 1];
    auto ptr               = std::lower_bound(
      split_proposals.ptr({feature_row_begin}), split_proposals.ptr(feature_row_end), x);
    if (ptr == split_proposals.ptr(feature_row_end)) return NOT_FOUND;
    return ptr - split_proposals.ptr(0);
  }
#endif

  __host__ __device__ std::tuple<int, int> FeatureRange(int feature) const
  {
    return std::make_tuple(row_pointers[feature], row_pointers[feature + 1]);
  }
};

class Histogram {
  legate::Buffer<GPair, 3> buffer_;  // Nodes, outputs, bins
  int node_begin_;
  int node_end_;
  std::size_t size_;

 public:
#ifdef __NVCC__
  Histogram(int node_begin, int node_end, int num_outputs, int num_bins, cudaStream_t stream)
    : node_begin_(node_begin), node_end_(node_end)
  {
    buffer_ = legate::create_buffer<GPair, 3>({node_end - node_begin, num_outputs, num_bins});
    size_   = (node_end - node_begin) * num_outputs * num_bins;
    CHECK_CUDA(
      cudaMemsetAsync(buffer_.ptr(legate::Point<3>::ZEROES()), 0, size_ * sizeof(GPair), stream));
  }
#else
  Histogram(int node_begin, int node_end, int num_outputs, int num_bins)
    : node_begin_(node_begin), node_end_(node_end)
  {
    buffer_ = legate::create_buffer<GPair, 3>({node_end - node_begin, num_outputs, num_bins});
    size_   = (node_end - node_begin) * num_outputs * num_bins;
    for (std::size_t i = 0; i < size_; i++) {
      buffer_.ptr(legate::Point<3>::ZEROES())[i] = GPair{0.0, 0.0};
    }
  }
#endif
  Histogram() = default;

  void Destroy()
  {
    if (size_ > 0) buffer_.destroy();
    node_begin_ = 0;
    node_end_   = 0;
    size_       = 0;
  }

  bool ContainsBatch(int node_begin_idx, int node_end_idx)
  {
    return node_begin_idx >= node_begin_ && node_end_idx <= node_end_;
  }

  __device__ bool ContainsNode(int node_idx)
  {
    return node_idx >= node_begin_ && node_idx < node_end_;
  }

  GPair* Ptr(int node_idx) { return buffer_.ptr({node_idx - node_begin_, 0, 0}); }

  std::size_t Size() { return size_; }

  // Node, output, bin
  __host__ __device__ GPair& operator[](legate::Point<3> p)
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

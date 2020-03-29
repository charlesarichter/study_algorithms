#include <iostream>

#include "conv_structs.hpp"

ConvKernels::ConvKernels(
    const std::vector<std::vector<Eigen::MatrixXd>>& kernels)
    : kernels_(kernels) {}

ConvKernels::ConvKernels(const std::vector<double>& weights,
                         const std::size_t num_kernels,
                         const std::size_t num_channels,
                         const std::size_t num_rows,
                         const std::size_t num_cols) {
  // Reshape weights into kernels.
  // std::vector<double>::const_iterator it = weights.begin();
  std::size_t ind = 0;
  for (std::size_t i = 0; i < num_kernels; ++i) {
    std::vector<Eigen::MatrixXd> kernel;
    for (std::size_t j = 0; j < num_channels; ++j) {
      kernel.emplace_back(Eigen::Map<const Eigen::MatrixXd>(
          weights.data() + ind, num_rows, num_cols));
      ind += num_rows * num_cols;
    }
    kernels_.emplace_back(kernel);
  }
}

std::vector<double> ConvKernels::GetWeights() const {
  std::vector<double> weights;
  for (const std::vector<Eigen::MatrixXd>& kernel : kernels_) {
    for (const Eigen::MatrixXd& channel : kernel) {
      weights.insert(weights.end(), channel.data(),
                     channel.data() + channel.rows() * channel.cols());
    }
  }
  return weights;
}

#pragma once

#include <eigen3/Eigen/Dense>
#include <vector>

/**
 * Parameters defining the convolution kernels of a convolution layer.
 *
 * TODO: Redesign this class to store the data as a std::vector<double> and use
 * Eigen::Map to reshape portions of the data into kernels and channels.
 */
class ConvKernels {
 public:
  ConvKernels(const std::vector<std::vector<Eigen::MatrixXd>>& kernels);
  ConvKernels(const std::vector<double>& weights, const std::size_t num_kernels,
              const std::size_t num_channels, const std::size_t num_rows,
              const std::size_t num_cols);

  const std::vector<std::vector<Eigen::MatrixXd>>& GetKernels() const {
    return kernels_;
  }

  std::vector<double> GetWeights() const;

  // Assume that kernels_ is populated and dimensionally consistent.
  std::size_t GetNumKernels() const { return kernels_.size(); }
  std::size_t GetNumChannels() const { return kernels_.front().size(); }
  std::size_t GetNumRows() const { return kernels_.front().front().rows(); }
  std::size_t GetNumCols() const { return kernels_.front().front().cols(); }

 private:
  std::vector<std::vector<Eigen::MatrixXd>> kernels_;

  // std::vector<double> weights;
  // std::size_t num_kernels;
  // std::size_t num_channels;
  // std::size_t num_rows;
  // std::size_t num_cols;
};

class InputOutputVolume {
 public:
  InputOutputVolume(const std::vector<Eigen::MatrixXd>& volume);

  std::vector<double> GetValues();

 private:
  std::vector<Eigen::MatrixXd> volume_;
};
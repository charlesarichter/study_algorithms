#include <iostream>
#include <random>

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

InputOutputVolume::InputOutputVolume() {}

InputOutputVolume::InputOutputVolume(const std::vector<Eigen::MatrixXd>& volume)
    : volume_(volume) {}

InputOutputVolume::InputOutputVolume(const std::vector<double>& values,
                                     const std::size_t num_channels,
                                     const std::size_t num_rows,
                                     const std::size_t num_cols) {
  // Reshape values into channels.
  std::size_t ind = 0;
  for (std::size_t i = 0; i < num_channels; ++i) {
    volume_.emplace_back(Eigen::Map<const Eigen::MatrixXd>(values.data() + ind,
                                                           num_rows, num_cols));
    ind += num_rows * num_cols;
  }
}

Eigen::VectorXd InputOutputVolume::GetVolumeVec() const {
  const std::vector<double> values = GetValues();  // TODO: Avoid this copy.
  return Eigen::Map<const Eigen::VectorXd>(values.data(), values.size());
}

std::vector<double> InputOutputVolume::GetValues() const {
  std::vector<double> values;
  for (const Eigen::MatrixXd& channel : volume_) {
    values.insert(values.end(), channel.data(),
                  channel.data() + channel.rows() * channel.cols());
  }
  return values;
}

ConvKernels GetRandomConvKernels(const std::size_t num_kernels,
                                 const std::size_t num_channels,
                                 const std::size_t num_rows,
                                 const std::size_t num_cols) {
  const std::size_t num_params =
      num_kernels * num_channels * num_rows * num_cols;
  std::vector<double> weights(num_params);
  std::default_random_engine generator;
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::generate(weights.begin(), weights.end(),
                [&dist, &generator]() { return dist(generator); });
  return ConvKernels(weights, num_kernels, num_channels, num_rows, num_cols);
}

InputOutputVolume GetRandomInputOutputVolume(const std::size_t num_channels,
                                             const std::size_t num_rows,
                                             const std::size_t num_cols) {
  const std::size_t num_params = num_channels * num_rows * num_cols;
  std::vector<double> values(num_params);
  std::default_random_engine generator;
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::generate(values.begin(), values.end(),
                [&dist, &generator]() { return dist(generator); });
  return InputOutputVolume(values, num_channels, num_rows, num_cols);
}

InputOutputVolume operator*(const InputOutputVolume& volume_1,
                            const InputOutputVolume& volume_2) {
  assert(volume_1.GetNumChannels() == volume_2.GetNumChannels());
  assert(volume_1.GetNumRows() == volume_2.GetNumRows());
  assert(volume_1.GetNumCols() == volume_2.GetNumCols());

  const std::vector<Eigen::MatrixXd>& v1 = volume_1.GetVolume();
  const std::vector<Eigen::MatrixXd>& v2 = volume_2.GetVolume();

  std::vector<Eigen::MatrixXd> v_result(v1.size());
  for (std::size_t i = 0; i < v1.size(); ++i) {
    v_result.at(i) = v1.at(i).cwiseProduct(v2.at(i));
  }

  return InputOutputVolume(v_result);
}

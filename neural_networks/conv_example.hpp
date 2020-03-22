#pragma once

#include <eigen3/Eigen/Dense>
#include <vector>

struct ConvExample {
  // Example input
  std::vector<Eigen::MatrixXd> input_volume;

  // Convolution kernels. Outer vector has size equal to number of kernels.
  // Inner vector has size equal to the number of channels of the input volume.
  std::vector<std::vector<Eigen::MatrixXd>> conv_kernels;

  // Biases. Number of biases is equal to the number of kernels.
  std::vector<double> biases;

  int stride;
  int padding;

  // Expected output
  std::vector<Eigen::MatrixXd> output_volume;
};

ConvExample GetConvExample1();
ConvExample GetConvExample2();
ConvExample GetConvExample3();

#pragma once

#include <eigen3/Eigen/Dense>
#include <vector>

void Conv(const std::vector<Eigen::MatrixXd>& input_volume,
          const std::vector<std::vector<Eigen::MatrixXd>>& conv_kernels,
          const std::vector<double>& biases,
          std::vector<Eigen::MatrixXd>* output_volume);

void TestConv();
void TestConv2();

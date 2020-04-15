#pragma once

#include "conv_example.hpp"
#include "conv_structs.hpp"

void RunConvTests();
void RunConvGradientTests();
void RunConvKernelTests();
void TestConv(const ConvExample& conv_example);
void TestConvGradient(const ConvExample& conv_example);
void TestConvKernels(const ConvExample& conv_example);

/**
 * Put together a full network with multiple layers.
 */
Eigen::VectorXd TestConvNet(
    const InputOutputVolume& input_volume, const ConvKernels& conv_kernels,
    const Eigen::MatrixXd& W_fc, const Eigen::VectorXd& b_fc,
    const Eigen::MatrixXd& W_out, const Eigen::VectorXd& b_out,
    std::vector<std::vector<Eigen::MatrixXd>>* manual_weight_gradients,
    std::vector<std::vector<Eigen::VectorXd>>* manual_bias_gradients);
void TestConvNetGradients();

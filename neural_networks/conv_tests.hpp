#pragma once

#include "conv_example.hpp"
#include "conv_structs.hpp"

void RunConvTests();
void RunConvGradientTests();
void RunConvKernelTests();
void TestConv(const ConvExample& conv_example);
void TestConvGradient(const ConvExample& conv_example);
void TestConvKernels(const ConvExample& conv_example);
void TestFullConv();

/**
 * Test setup with two conv layers. Work in progress...
 */
Eigen::VectorXd TestConvNetMultiConv(
    const InputOutputVolume& input_volume, const ConvKernels& conv_kernels_0,
    const std::vector<double>& conv_biases_0, const ConvKernels& conv_kernels_1,
    const std::vector<double>& conv_biases_1, const Eigen::MatrixXd& W2,
    const Eigen::VectorXd& b2, const Eigen::MatrixXd& W3,
    const Eigen::VectorXd& b3, const std::size_t num_steps_vertical_0,
    const std::size_t num_steps_horizontal_0,
    const std::size_t num_steps_vertical_1,
    const std::size_t num_steps_horizontal_1, const bool print,
    std::vector<Eigen::MatrixXd>* d_output_d_kernel,
    Eigen::VectorXd* d_output_d_bias, Eigen::MatrixXd* foo);

void TestConvNetGradientsMultiConv();

#pragma once

#include <eigen3/Eigen/Dense>
#include <vector>

/**
 * Input volume dimensions: Width (W) x Height (H) x Depth (D)
 *
 * Convolution kernels/filters have their own height and width, but their depth
 * must equal the depth of the input. And there K kernels/filters.
 *
 * After performing convolution, the output volume has a height and width
 * determined by the kernel width/height "spatial extent" (F), zero padding (P),
 * and stride (S).
 *
 * After performing convolution of the input with one filter, the results are
 * summed across channels and the flattened result becomes one depth slice of
 * the output volume.
 *
 * The depth of the output volume is equal to the number of kernels/filters: K.
 * That means the next layer will need to accept an input with K channels.
 *
 * The width of the output volume is:  W_out = (W - F + 2P)/S + 1.
 * The height of the output volume is: H_out = (H - F + 2P)/S + 1.
 * See http://cs231n.github.io/convolutional-networks/
 */
void Conv(const std::vector<Eigen::MatrixXd>& input_volume,
          const std::vector<std::vector<Eigen::MatrixXd>>& conv_kernels,
          const std::vector<double>& biases,
          std::vector<Eigen::MatrixXd>* output_volume);

void TestConv();
void TestConv2();

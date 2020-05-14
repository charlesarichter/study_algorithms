#pragma once

#include <eigen3/Eigen/Dense>
#include <vector>

#include "conv_example.hpp"
#include "conv_structs.hpp"

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
void Conv(const std::vector<Eigen::MatrixXd>& input_volume_unpadded,
          const std::vector<std::vector<Eigen::MatrixXd>>& conv_kernels,
          const std::vector<double>& biases, const int padding,
          const int stride, std::vector<Eigen::MatrixXd>* output_volume);

/**
 * Same as Conv() only implements the calculation as matrix multiplication.
 *
 * Helpful References on conversion between convolution/cross-correlation and
 * matrix multiplication:
 *
 * -https://knet.readthedocs.io/en/latest/cnn.html
 * -https://medium.com/@_init_/an-illustrated-explanation-of-performing-2d-convolutions-using-matrix-multiplications-1e8de8cd2544
 *
 * VERY helpful reference on computing gradients and backprop for conv layers:
 * - https://medium.com/@pavisj/convolutions-and-backpropagations-46026a8f5d2c
 *
 * Notes:
 * -Can copy filters into a sparse Toeplitz matrix
 * --Downside: Sparse matrix has a lot of zeros in it
 * --Downside: Need to constrain weights to have the same value in the various
 * places where they appear.
 * -Can copy input patches into a dense matrix.
 * --According to Knet reference above, this is the typical approach for 2D
 * images: "For 2-D images, typically the second approach is used: the local
 * patches of the image used by convolution are stretched out to columns of an
 * input matrix, an operation commonly called im2col. Each convolutional filter
 * is stretched out to rows of a filter matrix. After the matrix multiplication
 * the resulting array is reshaped into the proper output dimensions."
 *
 * ALSO (from knet): You just as you can perform convolution as matrix
 * multiplication, you can also perform matrix multiplication as convolution,
 * which is useful if you want to make a network layer that accepts inputs of
 * different sizes. Normal matrix multiplication would fail, but convolution
 * would work.
 */
void ConvMatrixMultiplication(
    const std::vector<Eigen::MatrixXd>& input_volume,
    const std::vector<std::vector<Eigen::MatrixXd>>& conv_kernels,
    const std::vector<double>& biases, const int padding, const int stride,
    std::vector<Eigen::MatrixXd>* output_volume,
    std::vector<Eigen::MatrixXd>* input_channels_unrolled_return);

/**
 * TODO: Documentation.
 */
std::vector<Eigen::MatrixXd> ConvGradient(
    const ConvKernels& conv_kernels,
    const std::vector<Eigen::MatrixXd>& next_grad);

std::vector<Eigen::MatrixXd> ConvWeightGradient(
    const std::vector<Eigen::MatrixXd>& input_volume,
    const std::vector<Eigen::MatrixXd>& next_grad);

/**
 * TODO: Enable different amount of horizontal and vertical padding.
 */
std::vector<Eigen::MatrixXd> PadVolume(
    const std::vector<Eigen::MatrixXd>& input_volume_unpadded, int padding);

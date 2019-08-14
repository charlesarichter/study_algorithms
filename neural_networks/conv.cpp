#include <iostream>

#include "conv.hpp"

void Conv(const std::vector<Eigen::MatrixXd>& input_volume,
          const std::vector<std::vector<Eigen::MatrixXd>>& conv_kernels,
          const std::vector<double>& biases,
          std::vector<Eigen::MatrixXd>* output_volume) {
  // TODO: Add padding and stride as inputs to this function.
  const size_t stride = 2;  // 1;

  // Get number of channels in the input volume.
  assert(!input_volume.empty());
  const size_t num_channels = input_volume.size();
  const size_t input_cols = input_volume.front().cols();  // NOTE: Padded.
  const size_t input_rows = input_volume.front().rows();  // NOTE: Padded.

  // Get filter size and confirm they are all the same.
  assert(!conv_kernels.empty());
  assert(!conv_kernels.front().empty());
  const size_t kernel_cols = conv_kernels.front().front().cols();
  const size_t kernel_rows = conv_kernels.front().front().rows();
  for (const std::vector<Eigen::MatrixXd>& f : conv_kernels) {
    assert(f.size() == num_channels);
  }
  assert(conv_kernels.size() == biases.size());

  // Determine number of horizontal/vertical steps.
  // TODO: Add an assert to catch cases where filter size plus stride doesn't
  // line up with the input dimensions.
  const size_t num_steps_horizontal = (input_cols - kernel_cols) / stride + 1;
  const size_t num_steps_vertical = (input_rows - kernel_rows) / stride + 1;

  for (size_t i = 0; i < conv_kernels.size(); ++i) {
    const std::vector<Eigen::MatrixXd>& conv_kernel = conv_kernels.at(i);

    // For each kernel/filter, sum results down the channels, plus bias.
    const double bias = biases.at(i);
    Eigen::MatrixXd filter_channel_sum =
        bias * Eigen::MatrixXd::Ones(num_steps_vertical, num_steps_horizontal);

    // Loop over channels of the input volume and filter. The depth (number of
    // channels) of the input volume must equal the depth (number of channels)
    // of each filter.
    for (size_t j = 0; j < conv_kernel.size(); ++j) {
      const Eigen::MatrixXd& input_channel = input_volume.at(j);
      const Eigen::MatrixXd& kernel_channel = conv_kernel.at(j);

      //   std::cerr << "Kernel channel:" << std::endl
      // << kernel_channel << std::endl;

      for (size_t k = 0; k < num_steps_horizontal; ++k) {
        const size_t min_ind_col = k * stride;
        for (size_t l = 0; l < num_steps_vertical; ++l) {
          const size_t min_ind_row = l * stride;

          // Extract sub-matrix we want to multiply.
          const Eigen::MatrixXd input_region = input_channel.block(
              min_ind_row, min_ind_col, kernel_rows, kernel_cols);

          // std::cerr << "Min row: " << min_ind_row << std::endl;
          // std::cerr << "Min col: " << min_ind_col << std::endl;
          // std::cerr << "l: " << l << std::endl;
          // std::cerr << "k: " << k << std::endl;
          // std::cerr << "input_region:" << std::endl
          //     << input_region << std::endl;
          // std::cin.get();

          filter_channel_sum(l, k) +=
              input_region.cwiseProduct(kernel_channel).sum();
          // std::cerr << "Filter Channel Sum:" << std::endl;
          // std::cerr << filter_channel_sum << std::endl;
          // std::cin.get();
        }
      }
    }

    std::cerr << "Filter Channel Sum:" << std::endl;
    std::cerr << filter_channel_sum << std::endl;
    std::cin.get();
  }
}

void TestConv() {
  // Test from example in: http://cs231n.github.io/convolutional-networks/

  // clang-format off

  // Input Volume (+pad 1) (7x7x3)
  // x[:,:,0]
  Eigen::MatrixXd x0(7, 7);
  x0 << 0, 0, 0, 0, 0, 0, 0, 
				0, 2, 2, 2, 2, 0, 0, 
				0, 1, 2, 1, 1, 1, 0, 
				0, 2, 1, 1, 1, 2, 0, 
				0, 2, 0, 2, 2, 0, 0, 
				0, 2, 0, 2, 0, 2, 0, 
				0, 0, 0, 0, 0, 0, 0;

  // x[:,:,1]
  Eigen::MatrixXd x1(7, 7);
  x1 << 0, 0, 0, 0, 0, 0, 0,
        0, 1, 1, 0, 2, 2, 0,
        0, 2, 1, 1, 0, 0, 0,
        0, 0, 0, 1, 1, 0, 0,
        0, 2, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 2, 0, 0,
        0, 0, 0, 0, 0, 0, 0;
    

  // x[:,:,2]
  Eigen::MatrixXd x2(7, 7);
  x2 << 0, 0, 0, 0, 0, 0, 0,
        0, 0, 2, 0, 1, 2, 0,
        0, 2, 1, 1, 2, 1, 0,
        0, 0, 2, 1, 0, 2, 0,
        0, 2, 2, 2, 1, 0, 0,
        0, 2, 2, 2, 2, 0, 0,
        0, 0, 0, 0, 0, 0, 0;


  // Filter W0 (3x3x3)
  // w0[:,:,0]
  Eigen::MatrixXd w00(3,3);
  w00 <<  0, -1, -1,
         -1,  0,  0, 
         -1, -1,  0;

  // w0[:,:,1],
  Eigen::MatrixXd w01(3,3);
  w01 << -1,  0, -1, 
          0,  1, -1, 
         -1, -1,  0;

  // w0[:,:,2],
  Eigen::MatrixXd w02(3,3);
  w02 <<  0,  0, -1, 
          1,  0,  0, 
         -1,  1,  1;

   // Bias b0 (1x1x1),
   // b0[:,:,0],
  // Eigen::MatrixXd b0(1,1);
  // b0 << 1;
  const double b0 = 1;

  // Filter W1 (3x3x3),
  // w1[:,:,0],
  Eigen::MatrixXd w10(3,3);
  w10 <<  1,  1,  0, 
          0, -1,  1, 
         -1, -1,  0;

  // w1[:,:,1],
  Eigen::MatrixXd w11(3,3);
  w11 <<  0,  1, -1, 
          0, -1, -1, 
          1,  0,  0;

  // w1[:,:,2],
  Eigen::MatrixXd w12(3,3);
  w12 <<  0, -1, -1, 
          1,  1,  1, 
         -1,  0,  1;

  // Bias b1 (1x1x1),
  // b1[:,:,0],
  // Eigen::MatrixXd b1(1,1);
  // b1 << 0;
  const double b1 = 0;

  // Output Volume (3x3x2),
  // o[:,:,0],
  Eigen::MatrixXd o0(3,3);
  o0 <<  1, -4, -1, 
        -4, -4, -4, 
        -2, -4,  3;

  // o[:,:,1],
  Eigen::MatrixXd o1(3,3);
  o1 <<  0,  0, -3, 
         0, -1, -2, 
         1,  1,  2;

  // clang-format on

  // Input volume is 3-channel.
  std::vector<Eigen::MatrixXd> input_volume{x0, x1, x2};

  // Each of the N (in ths case 2) conv filters must have as many channels as
  // the input (in this case 3).
  //
  // The depth of each conv filter must equal the depth of the input volume and
  // the number of conv filters determines the depth of the output volume.
  std::vector<std::vector<Eigen::MatrixXd>> conv_kernels{{w00, w01, w02},
                                                         {w10, w11, w12}};

  // TODO: Should biases be matrices? Will they always be 1x1?
  std::vector<double> biases{b0, b1};

  // Empty container for the output volume.
  std::vector<Eigen::MatrixXd> output_volume;

  // Compute conv layer.
  Conv(input_volume, conv_kernels, biases, &output_volume);
}

#include <iostream>

#include "conv.hpp"

void Conv(const std::vector<Eigen::MatrixXd>& input_volume_unpadded,
          const std::vector<std::vector<Eigen::MatrixXd>>& conv_kernels,
          const std::vector<double>& biases, const int padding,
          const int stride, std::vector<Eigen::MatrixXd>* output_volume) {
  // Get number of channels in the input volume.
  assert(!input_volume_unpadded.empty());
  const size_t num_channels = input_volume_unpadded.size();

  // Size of unpadded input.
  const size_t input_cols_unpadded = input_volume_unpadded.front().cols();
  const size_t input_rows_unpadded = input_volume_unpadded.front().rows();

  // Size of input after padding.
  const size_t input_cols_padded = input_cols_unpadded + 2 * padding;
  const size_t input_rows_padded = input_rows_unpadded + 2 * padding;

  // Create padded inputs.
  std::vector<Eigen::MatrixXd> input_volume;
  for (const Eigen::MatrixXd& input_channel_unpadded : input_volume_unpadded) {
    // Make a matrix of the padded size.
    Eigen::MatrixXd input_channel_padded =
        Eigen::MatrixXd::Zero(input_rows_padded, input_cols_padded);

    // Copy the unpadded input into the appropriate block of the padded input.
    input_channel_padded.block(padding, padding, input_rows_unpadded,
                               input_cols_unpadded) = input_channel_unpadded;
    input_volume.emplace_back(input_channel_padded);
  }

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

    // std::cerr << "Filter Channel Sum:" << std::endl;
    // std::cerr << filter_channel_sum << std::endl;
    // std::cin.get();
    output_volume->emplace_back(filter_channel_sum);
  }
}

void ConvMatrixMultiplication(
    const std::vector<Eigen::MatrixXd>& input_volume_unpadded,
    const std::vector<std::vector<Eigen::MatrixXd>>& conv_kernels,
    const std::vector<double>& biases, const int padding, const int stride,
    std::vector<Eigen::MatrixXd>* output_volume) {
  // Get number of channels in the input volume.
  assert(!input_volume_unpadded.empty());
  const size_t num_channels = input_volume_unpadded.size();

  // Size of unpadded input.
  const size_t input_cols_unpadded = input_volume_unpadded.front().cols();
  const size_t input_rows_unpadded = input_volume_unpadded.front().rows();

  // Size of input after padding.
  const size_t input_cols_padded = input_cols_unpadded + 2 * padding;
  const size_t input_rows_padded = input_rows_unpadded + 2 * padding;

  // Create padded inputs.
  std::vector<Eigen::MatrixXd> input_volume;
  for (const Eigen::MatrixXd& input_channel_unpadded : input_volume_unpadded) {
    // Make a matrix of the padded size.
    Eigen::MatrixXd input_channel_padded =
        Eigen::MatrixXd::Zero(input_rows_padded, input_cols_padded);

    // Copy the unpadded input into the appropriate block of the padded input.
    input_channel_padded.block(padding, padding, input_rows_unpadded,
                               input_cols_unpadded) = input_channel_unpadded;
    input_volume.emplace_back(input_channel_padded);
  }

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
  const size_t num_steps_total = num_steps_horizontal * num_steps_vertical;
  const size_t num_kernel_elements_total = kernel_cols * kernel_rows;

  std::cerr << "Num steps horizontal: " << num_steps_horizontal << std::endl;
  std::cerr << "Num steps vertical: " << num_steps_vertical << std::endl;
  std::cerr << "Num steps total: " << num_steps_total << std::endl;
  std::cerr << "Num kernel elements total: " << num_kernel_elements_total
            << std::endl;

  // // Copy unrolled patches of input into columns of a dense matrix.
  // // Unroll filters.
  //
  // for (const auto& i_mat : input_volume) {
  //   Eigen::MatrixXd i_non_const = i_mat;
  //   std::cerr << "Input channel:" << std::endl;
  //   std::cerr << i_non_const << std::endl;
  //   const Eigen::Map<Eigen::RowVectorXd> i_map(i_non_const.data(),
  //                                              i_non_const.size());
  //   std::cerr << "Input unrolled:" << std::endl;
  //   std::cerr << i_map << std::endl;
  // }
  //
  // for (const auto& k_vec : conv_kernels) {
  //   for (const auto& k : k_vec) {
  //     Eigen::MatrixXd k_non_const = k;
  //     std::cerr << "Kernel:" << std::endl;
  //     std::cerr << k_non_const << std::endl;
  //     const Eigen::Map<Eigen::RowVectorXd> k_map(k_non_const.data(),
  //                                                k_non_const.size());
  //     std::cerr << "Kernel unrolled:" << std::endl;
  //     std::cerr << k_map << std::endl;
  //   }
  // }

  for (size_t i = 0; i < conv_kernels.size(); ++i) {
    const std::vector<Eigen::MatrixXd>& conv_kernel = conv_kernels.at(i);

    // For each kernel/filter, sum results down the channels, plus bias.
    // const double bias = biases.at(i);
    // Eigen::MatrixXd filter_channel_sum =
    //     bias * Eigen::MatrixXd::Ones(num_steps_vertical,
    //     num_steps_horizontal);

    Eigen::MatrixXd input_patches_unrolled =
        Eigen::MatrixXd::Zero(num_kernel_elements_total, num_steps_total);

    // Loop over channels of the input volume and filter. The depth (number of
    // channels) of the input volume must equal the depth (number of channels)
    // of each filter.
    for (size_t j = 0; j < conv_kernel.size(); ++j) {
      const Eigen::MatrixXd& input_channel = input_volume.at(j);
      const Eigen::MatrixXd& kernel_channel = conv_kernel.at(j);

      // std::cerr << "Kernel channel:" << std::endl
      //           << kernel_channel << std::endl;

      // TODO: Improve this. Using non-const so that we can call .data().
      Eigen::MatrixXd kernel_channel_non_const = kernel_channel;
      const Eigen::Map<Eigen::VectorXd> kernel_unrolled(
          kernel_channel_non_const.data(), kernel_channel_non_const.size());
      std::cerr << "Kernel unrolled: " << std::endl
                << kernel_unrolled << std::endl;

      // TODO: Invert the order of some of these loops so that we build
      // input_patches_unrolled once and we can use it for multiple kernels.
      // First determine the size and number of input patches (must be same for
      // all kernels). Then build input_patches_unrolled. Then loop over kernels
      // and apply them. The loop structure as-is will build the same
      // input_patches_unrolled for each kernel.
      std::size_t input_patch = 0;
      for (size_t k = 0; k < num_steps_horizontal; ++k) {
        const size_t min_ind_col = k * stride;
        for (size_t l = 0; l < num_steps_vertical; ++l) {
          const size_t min_ind_row = l * stride;

          // Extract sub-matrix we want to multiply.
          Eigen::MatrixXd input_region = input_channel.block(
              min_ind_row, min_ind_col, kernel_rows, kernel_cols);

          // Reshape this sub-matrix into a vector.
          const Eigen::Map<Eigen::VectorXd> input_region_vec(
              input_region.data(), input_region.size());

          // std::cerr << "input region vec: " << std::endl
          //           << input_region_vec << std::endl;
          // std::cerr << "input patch: " << input_patch << std::endl;

          input_patches_unrolled.col(input_patch) << input_region_vec;
          // std::cerr << "input patches unrolled: " << std::endl
          //           << input_patches_unrolled << std::endl;

          // TODO: Compute this from k and l rather than incrementing.
          ++input_patch;
        }
      }

      std::cerr << "input patches unrolled: " << std::endl
                << input_patches_unrolled << std::endl;

      Eigen::VectorXd conv_result_unrolled =
          kernel_unrolled.transpose() * input_patches_unrolled;
      std::cerr << "Convolution result unrolled: " << std::endl
                << conv_result_unrolled << std::endl;

      const Eigen::MatrixXd conv_result =
          Eigen::Map<Eigen::MatrixXd>(conv_result_unrolled.data(),
                                      num_steps_horizontal, num_steps_vertical);
      std::cerr << "Convolution result: " << std::endl
                << conv_result << std::endl;

      output_volume->emplace_back(conv_result);
    }
  }
}

void TestConv() {
  // Test from example in: http://cs231n.github.io/convolutional-networks/

  // clang-format off

  // Input Volume (+pad 1) (7x7x3)
  // x[:,:,0]
  // Eigen::MatrixXd x0(7, 7);
  // x0 << 0, 0, 0, 0, 0, 0, 0,
	// 			0, 2, 2, 2, 2, 0, 0,
	// 			0, 1, 2, 1, 1, 1, 0,
	// 			0, 2, 1, 1, 1, 2, 0,
	// 			0, 2, 0, 2, 2, 0, 0,
	// 			0, 2, 0, 2, 0, 2, 0,
	// 			0, 0, 0, 0, 0, 0, 0;
  //
  // // x[:,:,1]
  // Eigen::MatrixXd x1(7, 7);
  // x1 << 0, 0, 0, 0, 0, 0, 0,
  //       0, 1, 1, 0, 2, 2, 0,
  //       0, 2, 1, 1, 0, 0, 0,
  //       0, 0, 0, 1, 1, 0, 0,
  //       0, 2, 0, 0, 0, 0, 0,
  //       0, 1, 0, 0, 2, 0, 0,
  //       0, 0, 0, 0, 0, 0, 0;
  //
  // // x[:,:,2]
  // Eigen::MatrixXd x2(7, 7);
  // x2 << 0, 0, 0, 0, 0, 0, 0,
  //       0, 0, 2, 0, 1, 2, 0,
  //       0, 2, 1, 1, 2, 1, 0,
  //       0, 0, 2, 1, 0, 2, 0,
  //       0, 2, 2, 2, 1, 0, 0,
  //       0, 2, 2, 2, 2, 0, 0,
  //       0, 0, 0, 0, 0, 0, 0;

  // Input volume without padding (to be added in the Conv function).
  // x[:,:,0]
  Eigen::MatrixXd x0(5, 5);
	x0 << 2, 2, 2, 2, 0,
				1, 2, 1, 1, 1,
				2, 1, 1, 1, 2,
				2, 0, 2, 2, 0,
				2, 0, 2, 0, 2;

  // x[:,:,1]
  Eigen::MatrixXd x1(5, 5);
  x1 << 1, 1, 0, 2, 2,
        2, 1, 1, 0, 0,
        0, 0, 1, 1, 0,
        2, 0, 0, 0, 0,
        1, 0, 0, 2, 0;

  // x[:,:,2]
  Eigen::MatrixXd x2(5, 5);
  x2 << 0, 2, 0, 1, 2,
        2, 1, 1, 2, 1,
        0, 2, 1, 0, 2,
        2, 2, 2, 1, 0,
        2, 2, 2, 2, 0;

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

  const std::vector<Eigen::MatrixXd> output_volume_expected{o0, o1};

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

  // Padding of 1 for this example.
  const int padding = 1;

  // Stride of 2 for this example.
  const int stride = 2;

  // Compute conv layer.
  Conv(input_volume, conv_kernels, biases, padding, stride, &output_volume);

  assert(output_volume.size() == output_volume_expected.size());

  for (std::size_t i = 0; i < output_volume.size(); ++i) {
    const Eigen::MatrixXd& output_computed = output_volume.at(i);
    const Eigen::MatrixXd& output_expected = output_volume_expected.at(i);
    const Eigen::MatrixXd output_diff = output_expected - output_computed;
    std::cerr << "Output difference: " << std::endl << output_diff << std::endl;
  }
}

void TestConv2() {
  // Test from example in: http://cs231n.github.io/convolutional-networks/

  // clang-format off

  // Input Volume (+pad 1) (7x7x3)
  // x[:,:,0]
  // Eigen::MatrixXd x0(7, 7);
  // x0 << 0, 0, 0, 0, 0, 0, 0,
	// 			0, 1, 2, 2, 0, 1, 0,
	// 			0, 0, 0, 2, 2, 0, 0,
	// 			0, 0, 1, 1, 2, 2, 0,
	// 			0, 1, 1, 2, 1, 1, 0,
	// 			0, 0, 0, 2, 2, 1, 0,
	// 			0, 0, 0, 0, 0, 0, 0;
  //
  // // x[:,:,1]
  // Eigen::MatrixXd x1(7, 7);
  // x1 << 0, 0, 0, 0, 0, 0, 0,
  //       0, 2, 0, 2, 0, 1, 0,
  //       0, 2, 2, 2, 2, 2, 0,
  //       0, 1, 2, 2, 1, 1, 0,
  //       0, 1, 0, 1, 1, 1, 0,
  //       0, 2, 0, 1, 2, 0, 0,
  //       0, 0, 0, 0, 0, 0, 0;
  //
  // // x[:,:,2]
  // Eigen::MatrixXd x2(7, 7);
  // x2 << 0, 0, 0, 0, 0, 0, 0,
  //       0, 2, 2, 0, 2, 2, 0,
  //       0, 0, 1, 2, 1, 1, 0,
  //       0, 2, 2, 1, 1, 0, 0,
  //       0, 0, 1, 0, 2, 2, 0,
  //       0, 2, 1, 2, 0, 1, 0,
  //       0, 0, 0, 0, 0, 0, 0;

  // Input volume without padding (to be added in the Conv function).
  // x[:,:,0]
  Eigen::MatrixXd x0(5, 5);
  x0 << 1, 2, 2, 0, 1,
				0, 0, 2, 2, 0,
				0, 1, 1, 2, 2,
				1, 1, 2, 1, 1,
				0, 0, 2, 2, 1;

  // x[:,:,1]
  Eigen::MatrixXd x1(5, 5);
  x1 << 2, 0, 2, 0, 1,
        2, 2, 2, 2, 2,
        1, 2, 2, 1, 1,
        1, 0, 1, 1, 1,
        2, 0, 1, 2, 0;

  // x[:,:,2]
  Eigen::MatrixXd x2(5, 5);
  x2 << 2, 2, 0, 2, 2,
        0, 1, 2, 1, 1,
        2, 2, 1, 1, 0,
        0, 1, 0, 2, 2,
        2, 1, 2, 0, 1;

  // Filter W0 (3x3x3)
  // w0[:,:,0]
  Eigen::MatrixXd w00(3,3);
  w00 << -1,  1, -1,
          0,  0, -1,
         -1,  1, -1;

  // w0[:,:,1],
  Eigen::MatrixXd w01(3,3);
  w01 <<  0,  0,  1,
         -1,  1,  1,
          1,  0,  0;

  // w0[:,:,2],
  Eigen::MatrixXd w02(3,3);
  w02 << -1,  0, -1,
          0,  0,  0,
         -1, -1, -1;

   // Bias b0 (1x1x1),
   // b0[:,:,0],
  // Eigen::MatrixXd b0(1,1);
  // b0 << 1;
  const double b0 = 1;

  // Filter W1 (3x3x3),
  // w1[:,:,0],
  Eigen::MatrixXd w10(3,3);
  w10 <<  1,  1,  0,
          0,  0, -1,
          0, -1, -1;

  // w1[:,:,1],
  Eigen::MatrixXd w11(3,3);
  w11 << -1, -1, -1,
          1,  0,  1,
         -1, -1,  1;

  // w1[:,:,2],
  Eigen::MatrixXd w12(3,3);
  w12 << -1,  0,  0,
          0,  0,  0,
          0,  1,  1;

  // Bias b1 (1x1x1),
  // b1[:,:,0],
  // Eigen::MatrixXd b1(1,1);
  // b1 << 0;
  const double b1 = 0;

  // Output Volume (3x3x2),
  // o[:,:,0],
  Eigen::MatrixXd o0(3,3);
  o0 <<  0,  1,  0,
         3, -3, -5,
         2,  0, -3;

  // o[:,:,1],
  Eigen::MatrixXd o1(3,3);
  o1 << -1, -3, -3,
        -5, -5, -3,
         0,  0,  0;

  // clang-format on

  const std::vector<Eigen::MatrixXd> output_volume_expected{o0, o1};

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

  // Padding of 1 for this example.
  const int padding = 1;

  // Stride of 2 for this example.
  const double stride = 2;

  // Compute conv layer.
  Conv(input_volume, conv_kernels, biases, padding, stride, &output_volume);

  assert(output_volume.size() == output_volume_expected.size());

  for (std::size_t i = 0; i < output_volume.size(); ++i) {
    const Eigen::MatrixXd& output_computed = output_volume.at(i);
    const Eigen::MatrixXd& output_expected = output_volume_expected.at(i);
    const Eigen::MatrixXd output_diff = output_expected - output_computed;
    std::cerr << "Output difference: " << std::endl << output_diff << std::endl;
  }
}

void TestConv3knet() {
  // clang-format off

  // Input Volume (+pad 0) (3x3x1)
  // x[:,:,0]
  Eigen::MatrixXd x0(3, 3);
  x0 << 1, 4, 7,
				2, 5, 8,
				3, 6, 9;

  // Filter W0 (2x2x1)
  // w0[:,:,0]
  Eigen::MatrixXd w00(2,2);
  // w00 <<  1,  3,
  //         2,  4;
  w00 <<  4,  2,
          3,  1;

   // Bias b0 (1x1x1),
   // b0[:,:,0],
  // Eigen::MatrixXd b0(1,1);
  // b0 << 1;
  const double b0 = 0;

  // clang-format on

  // Input volume is 3-channel.
  std::vector<Eigen::MatrixXd> input_volume{x0};

  // Each of the N (in ths case 2) conv filters must have as many channels as
  // the input (in this case 3).
  //
  // The depth of each conv filter must equal the depth of the input volume and
  // the number of conv filters determines the depth of the output volume.
  std::vector<std::vector<Eigen::MatrixXd>> conv_kernels{{w00}};

  // TODO: Should biases be matrices? Will they always be 1x1?
  std::vector<double> biases{b0};

  // Empty container for the output volume.
  std::vector<Eigen::MatrixXd> output_volume;
  std::vector<Eigen::MatrixXd> output_volume_mat_mult;

  // Padding of 0 for this example.
  const int padding = 0;

  // Stride of 1 for this example.
  const double stride = 1;

  // Compute conv layer.
  Conv(input_volume, conv_kernels, biases, padding, stride, &output_volume);
  ConvMatrixMultiplication(input_volume, conv_kernels, biases, padding, stride,
                           &output_volume_mat_mult);

  std::cerr << "Size of output volume regular  " << output_volume.size()
            << std::endl;
  std::cerr << "Output volume regular  " << std::endl
            << output_volume.front() << std::endl;

  std::cerr << "Size of output volume mat mult "
            << output_volume_mat_mult.size() << std::endl;
  std::cerr << "Output volume mat mult  " << std::endl
            << output_volume_mat_mult.front() << std::endl;
}

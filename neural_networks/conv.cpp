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

  // std::cerr << "Num steps horizontal: " << num_steps_horizontal << std::endl;
  // std::cerr << "Num steps vertical: " << num_steps_vertical << std::endl;
  // std::cerr << "Num steps total: " << num_steps_total << std::endl;
  // std::cerr << "Num kernel elements total: " << num_kernel_elements_total
  //           << std::endl;

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
    const double bias = biases.at(i);
    Eigen::MatrixXd filter_channel_sum =
        bias * Eigen::MatrixXd::Ones(num_steps_vertical, num_steps_horizontal);

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
      // std::cerr << "Kernel unrolled: " << std::endl
      //           << kernel_unrolled << std::endl;

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

      // std::cerr << "input patches unrolled: " << std::endl
      //           << input_patches_unrolled << std::endl;

      Eigen::VectorXd conv_result_unrolled =
          kernel_unrolled.transpose() * input_patches_unrolled;
      // std::cerr << "Convolution result unrolled: " << std::endl
      //           << conv_result_unrolled << std::endl;

      const Eigen::MatrixXd conv_result =
          Eigen::Map<Eigen::MatrixXd>(conv_result_unrolled.data(),
                                      num_steps_horizontal, num_steps_vertical);
      // std::cerr << "Convolution result: " << std::endl
      //           << conv_result << std::endl;

      filter_channel_sum += conv_result;
    }
    output_volume->emplace_back(filter_channel_sum);
  }
}

void TestConv(const ConvExample& conv_example) {
  const std::vector<Eigen::MatrixXd>& output_volume_expected =
      conv_example.output_volume;
  const std::vector<Eigen::MatrixXd>& input_volume = conv_example.input_volume;
  const std::vector<std::vector<Eigen::MatrixXd>>& conv_kernels =
      conv_example.conv_kernels;
  const std::vector<double>& biases = conv_example.biases;
  const int padding = conv_example.padding;
  const int stride = conv_example.stride;

  // Empty container for the output volume.
  std::vector<Eigen::MatrixXd> output_volume;
  std::vector<Eigen::MatrixXd> output_volume_mat_mult;

  // Compute conv layer.
  Conv(input_volume, conv_kernels, biases, padding, stride, &output_volume);
  ConvMatrixMultiplication(input_volume, conv_kernels, biases, padding, stride,
                           &output_volume_mat_mult);

  assert(output_volume.size() == output_volume_expected.size());
  assert(output_volume_mat_mult.size() == output_volume_expected.size());

  for (std::size_t i = 0; i < output_volume.size(); ++i) {
    const Eigen::MatrixXd& output_expected = output_volume_expected.at(i);
    const Eigen::MatrixXd& output_computed = output_volume.at(i);
    const Eigen::MatrixXd output_diff = output_expected - output_volume.at(i);
    std::cerr << "Output difference: " << std::endl << output_diff << std::endl;

    const Eigen::MatrixXd& output_computed_mat_mult =
        output_volume_mat_mult.at(i);
    const Eigen::MatrixXd output_diff_mat_mult =
        output_expected - output_volume_mat_mult.at(i);
    std::cerr << "Output difference matrix multiplication: " << std::endl
              << output_diff_mat_mult << std::endl;
  }
}

void TestConvGradient(const ConvExample& conv_example) {
  const std::vector<Eigen::MatrixXd>& output_volume_expected =
      conv_example.output_volume;
  const std::vector<Eigen::MatrixXd>& input_volume = conv_example.input_volume;
  const std::vector<std::vector<Eigen::MatrixXd>>& conv_kernels =
      conv_example.conv_kernels;
  const std::vector<double>& biases = conv_example.biases;
  const int padding = conv_example.padding;
  const int stride = conv_example.stride;

  // Compute nominal output.
  std::vector<Eigen::MatrixXd> output_volume;
  ConvMatrixMultiplication(input_volume, conv_kernels, biases, padding, stride,
                           &output_volume);

  const double delta = 1e-6;

  // Loop over kernels.
  for (std::size_t i = 0; i < conv_kernels.size(); ++i) {
    // Loop over channels of each kernel.
    for (std::size_t j = 0; j < conv_kernels.at(i).size(); ++j) {
      // Loop over parameters of each channel of each kernel.
      for (std::size_t k = 0; k < conv_kernels.at(i).at(j).rows(); ++k) {
        for (std::size_t l = 0; l < conv_kernels.at(i).at(j).cols(); ++l) {
          // Copy the nominal kernels.
          std::vector<std::vector<Eigen::MatrixXd>> conv_kernels_plus =
              conv_kernels;

          // Add delta perturbation to specific parameter.
          conv_kernels_plus.at(i).at(j)(k, l) += delta;

          // Evaluate output with perturbed kernel.
          std::vector<Eigen::MatrixXd> output_volume_plus;
          ConvMatrixMultiplication(input_volume, conv_kernels_plus, biases,
                                   padding, stride, &output_volume_plus);
          std::cerr << i << " " << j << std::endl;
        }
      }
    }
  }
}

void TestConvKernels(const ConvExample& conv_example) {
  // Construct ConvKernels using vector of vector of matrices.
  const ConvKernels ck(conv_example.conv_kernels);

  // Unpack weights and dimensions.
  const std::vector<double> weights = ck.GetWeights();
  const std::size_t num_kernels = ck.GetNumKernels();
  const std::size_t num_channels = ck.GetNumChannels();
  const std::size_t num_rows = ck.GetNumRows();
  const std::size_t num_cols = ck.GetNumCols();

  // Construct ConvKernels using weights and dimensions.
  const ConvKernels ck_from_weights(weights, num_kernels, num_channels,
                                    num_rows, num_cols);

  // Get kernels and make sure they are equal to the original.
  const std::vector<std::vector<Eigen::MatrixXd>>& kernels_original =
      ck.GetKernels();
  const std::vector<std::vector<Eigen::MatrixXd>>& kernels_reconstructed =
      ck_from_weights.GetKernels();

  assert(kernels_original.size() == kernels_reconstructed.size());
  for (std::size_t i = 0; i < kernels_original.size(); ++i) {
    const std::vector<Eigen::MatrixXd>& kernel_original =
        kernels_original.at(i);
    const std::vector<Eigen::MatrixXd>& kernel_reconstructed =
        kernels_reconstructed.at(i);
    assert(kernel_original.size() == kernel_reconstructed.size());
    for (std::size_t j = 0; j < kernel_original.size(); ++j) {
      const Eigen::MatrixXd& channel_original = kernel_original.at(j);
      const Eigen::MatrixXd& channel_reconstructed = kernel_reconstructed.at(j);
      assert(channel_original.rows() == channel_reconstructed.rows());
      assert(channel_original.cols() == channel_reconstructed.cols());

      // Confirm equality.
      const Eigen::MatrixXd channel_difference =
          channel_original - channel_reconstructed;
      assert(channel_difference.cwiseAbs().maxCoeff() < 1e-6);
    }
  }
}

void RunConvTests() {
  const ConvExample conv_example_1 = GetConvExample1();
  TestConv(conv_example_1);
  const ConvExample conv_example_2 = GetConvExample2();
  TestConv(conv_example_2);
  const ConvExample conv_example_3 = GetConvExample3();
  TestConv(conv_example_3);
}

void RunConvGradientTests() {
  const ConvExample conv_example_1 = GetConvExample1();
  TestConvGradient(conv_example_1);
}

void RunConvKernelTests() {
  const ConvExample conv_example_1 = GetConvExample1();
  TestConvKernels(conv_example_1);
}

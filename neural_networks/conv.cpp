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

  // Convert input channels to their "unrolled" form so that each column of each
  // channel represents the set of elements that will be multiplied by the
  // kernel placed in a certain location.
  std::vector<Eigen::MatrixXd> input_channels_unrolled;
  for (size_t j = 0; j < num_channels; ++j) {
    const Eigen::MatrixXd& input_channel = input_volume.at(j);
    Eigen::MatrixXd input_channel_unrolled =
        Eigen::MatrixXd::Zero(num_kernel_elements_total, num_steps_total);
    for (size_t k = 0; k < num_steps_horizontal; ++k) {
      const size_t min_ind_col = k * stride;
      for (size_t l = 0; l < num_steps_vertical; ++l) {
        const size_t min_ind_row = l * stride;

        // Extract sub-matrix we want to multiply.
        Eigen::MatrixXd input_region = input_channel.block(
            min_ind_row, min_ind_col, kernel_rows, kernel_cols);
        const std::size_t ind = l + k * num_steps_vertical;  // 0, 1, 2, 3,...
        input_channel_unrolled.col(ind) = Eigen::Map<Eigen::VectorXd>(
            input_region.data(), input_region.size());
      }
    }
    input_channels_unrolled.emplace_back(input_channel_unrolled);
  }

  // Convert kernels to their unrolled form.
  std::vector<std::vector<Eigen::VectorXd>> conv_kernels_unrolled;
  for (size_t i = 0; i < conv_kernels.size(); ++i) {
    std::vector<Eigen::VectorXd> conv_kernel_unrolled;
    const std::vector<Eigen::MatrixXd>& conv_kernel = conv_kernels.at(i);
    for (size_t j = 0; j < conv_kernel.size(); ++j) {
      const Eigen::MatrixXd& kernel_channel = conv_kernel.at(j);
      conv_kernel_unrolled.emplace_back(Eigen::Map<const Eigen::VectorXd>(
          kernel_channel.data(), kernel_channel.size()));
    }
    conv_kernels_unrolled.emplace_back(conv_kernel_unrolled);
  }

  // Compute convolution layer.
  for (size_t i = 0; i < conv_kernels.size(); ++i) {
    // For each kernel/filter, sum results down the channels, plus bias.
    const double bias = biases.at(i);
    Eigen::MatrixXd filter_channel_sum =
        bias * Eigen::MatrixXd::Ones(num_steps_vertical, num_steps_horizontal);

    const std::vector<Eigen::VectorXd>& conv_kernel_unrolled =
        conv_kernels_unrolled.at(i);

    // Loop over channels of the input volume and filter. The depth (number of
    // channels) of the input volume must equal the depth (number of channels)
    // of each filter.
    for (size_t j = 0; j < num_channels; ++j) {
      const Eigen::VectorXd& conv_kernel_channel_unrolled =
          conv_kernel_unrolled.at(j);
      const Eigen::MatrixXd& input_channel_unrolled =
          input_channels_unrolled.at(j);

      // Here is the actual multiplication between kernel and input values.
      Eigen::VectorXd conv_result_unrolled =
          conv_kernel_channel_unrolled.transpose() * input_channel_unrolled;

      std::cerr << "Gradient of output w.r.t. weights: " << std::endl
                << input_channel_unrolled << std::endl;

      // Reshape into the dimensions of the filter channel sum.
      const Eigen::MatrixXd conv_result =
          Eigen::Map<Eigen::MatrixXd>(conv_result_unrolled.data(),
                                      num_steps_horizontal, num_steps_vertical);
      filter_channel_sum += conv_result;
    }
    output_volume->emplace_back(filter_channel_sum);
  }
}

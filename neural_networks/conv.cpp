#include <iostream>

#include "conv.hpp"

void Conv(const std::vector<Eigen::MatrixXd>& input_volume_unpadded,
          const std::vector<std::vector<Eigen::MatrixXd>>& conv_kernels,
          const std::vector<double>& biases, const int padding,
          const int stride, std::vector<Eigen::MatrixXd>* output_volume) {
  // Get number of channels in the input volume.
  assert(!input_volume_unpadded.empty());
  const size_t num_channels = input_volume_unpadded.size();

  // Pad input.
  const std::vector<Eigen::MatrixXd> input_volume =
      PadVolume(input_volume_unpadded, padding);

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
      for (size_t k = 0; k < num_steps_horizontal; ++k) {
        const size_t min_ind_col = k * stride;
        for (size_t l = 0; l < num_steps_vertical; ++l) {
          const size_t min_ind_row = l * stride;

          // Extract sub-matrix we want to multiply.
          // NOTE: This line takes about 60% of this function's time.
          const Eigen::MatrixXd& input_region = input_channel.block(
              min_ind_row, min_ind_col, kernel_rows, kernel_cols);

          // NOTE: This line takes about 50% of this function's time.
          filter_channel_sum(l, k) +=
              input_region.cwiseProduct(kernel_channel).sum();
        }
      }
    }
    output_volume->emplace_back(filter_channel_sum);
  }
}

void ConvMatrixMultiplication(
    const std::vector<Eigen::MatrixXd>& input_volume_unpadded,
    const std::vector<std::vector<Eigen::MatrixXd>>& conv_kernels,
    const std::vector<double>& biases, const int padding, const int stride,
    std::vector<Eigen::MatrixXd>* output_volume,
    std::vector<Eigen::MatrixXd>* input_channels_unrolled_return) {
  // Get number of channels in the input volume.
  assert(!input_volume_unpadded.empty());
  const size_t num_channels = input_volume_unpadded.size();

  // Pad input.
  const std::vector<Eigen::MatrixXd> input_volume =
      PadVolume(input_volume_unpadded, padding);

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

  // Convert input channels to their "unrolled" form so that each column of each
  // channel represents the set of elements that will be multiplied by the
  // kernel placed in a certain location.
  // const std::vector<Eigen::MatrixXd> input_channels_unrolled =
  //     BuildConvInputMatrix(input_volume, kernel_rows, kernel_cols, stride);
  const std::vector<Eigen::MatrixXd> input_channels_unrolled =
      BuildConvInputMatrixElementWise(input_volume, kernel_rows, kernel_cols,
                                      stride);

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

      // std::cerr << "Gradient of output w.r.t. weights: " << std::endl
      //           << input_channel_unrolled << std::endl;

      // Reshape into the dimensions of the filter channel sum.
      const Eigen::MatrixXd conv_result =
          Eigen::Map<Eigen::MatrixXd>(conv_result_unrolled.data(),
                                      num_steps_vertical, num_steps_horizontal);
      filter_channel_sum += conv_result;
    }
    output_volume->emplace_back(filter_channel_sum);
  }

  // Copy unrolled input channels to the output.
  // TODO: Find a better way to do this, or better yet, just return gradient.
  *input_channels_unrolled_return = input_channels_unrolled;
}

std::vector<Eigen::MatrixXd> ConvGradient(
    const ConvKernels& conv_kernels,
    const std::vector<Eigen::MatrixXd>& next_grad) {
  const std::vector<std::vector<Eigen::MatrixXd>>& kernels =
      conv_kernels.GetKernels();
  const std::size_t num_kernels = kernels.size();
  const std::size_t num_channels_per_kernel = kernels.front().size();

  std::vector<Eigen::MatrixXd> output_volume;

  // For each kernel in conv_kernels, perform convolution.
  for (std::size_t j = 0; j < num_channels_per_kernel; ++j) {
    std::vector<Eigen::MatrixXd> input_volume;
    for (std::size_t i = 0; i < num_kernels; ++i) {
      // Unpack the jth channel of the ith kernel.
      const Eigen::MatrixXd& kernel_channel = kernels.at(i).at(j);
      input_volume.emplace_back(
          kernel_channel.rowwise().reverse().colwise().reverse());
    }
    assert(next_grad.size() == input_volume.size());

    // TODO: Currently only works with square kernels and inputs due to the
    // padding required for a full convolution.
    const std::size_t conv_kernels_rows = next_grad.front().rows();
    const std::size_t conv_kernels_cols = next_grad.front().cols();

    // NOTE: "Full convolution" involves sweeping the filter all the way
    // across, max possible overlap, which can be achieved by doing a
    // "normal" convolution with a padded input.
    std::vector<Eigen::MatrixXd> input_channels_unrolled_return;
    std::vector<Eigen::MatrixXd> output_volume_iteration;
    const std::size_t full_conv_padding = conv_kernels_rows - 1;
    ConvMatrixMultiplication(
        input_volume, std::vector<std::vector<Eigen::MatrixXd>>{next_grad}, {0},
        full_conv_padding, 1, &output_volume_iteration,
        &input_channels_unrolled_return);

    // Is it true that this will always be a single "channel" output?
    assert(output_volume_iteration.size() == 1);
    output_volume.emplace_back(output_volume_iteration.front()
                                   .rowwise()
                                   .reverse()
                                   .colwise()
                                   .reverse());
  }
  return output_volume;
}

std::vector<Eigen::MatrixXd> ConvWeightGradient(
    const std::vector<Eigen::MatrixXd>& input_volume,
    const std::vector<Eigen::MatrixXd>& next_grad) {
  // What we are doing here is individually convolving each channel of
  // input_volume with each "channel" of next_grad.
  //
  // General note for here and elsewhere: See if we can avoid looping over
  // kernels if we can compute convolution with each kernel simultaneously via
  // matrix multiplication, whether the unrolled weight/kernel vector is a
  // matrix of unrolled kernels stacked together.
  //
  // This same operation can also be accomplished as follows, using the unrolled
  // input matrix used for convolution in the forward pass, and next_grad
  // reshaped into a matrix where each colum contains the weights of a kernel.
  // Don't forget that this is a convolution!
  //
  // for (std::size_t i = 0; i < num_kernels_0; ++i) {
  //   dydw_kernels.emplace_back(conv_0_input_mat.at(i) * dydl0_wrapped);
  // }
  std::vector<Eigen::MatrixXd> dydw_kernels;
  for (int j = 0; j < next_grad.size(); ++j) {
    for (int i = 0; i < input_volume.size(); ++i) {
      std::vector<Eigen::MatrixXd> output_volume_iteration;
      std::vector<Eigen::MatrixXd> input_channels_unrolled_return;
      ConvMatrixMultiplication({input_volume.at(i)}, {{next_grad.at(j)}}, {0},
                               0, 1, &output_volume_iteration,
                               &input_channels_unrolled_return);
      assert(output_volume_iteration.size() == 1);
      for (const auto& o : output_volume_iteration) {
        dydw_kernels.emplace_back(o);
      }
    }
  }
  return dydw_kernels;
}

std::vector<Eigen::MatrixXd> PadVolume(
    const std::vector<Eigen::MatrixXd>& input_volume_unpadded, int padding) {
  const size_t input_cols_unpadded = input_volume_unpadded.front().cols();
  const size_t input_rows_unpadded = input_volume_unpadded.front().rows();
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
  return input_volume;
}

std::vector<Eigen::MatrixXd> BuildConvInputMatrix(
    const std::vector<Eigen::MatrixXd>& input_volume, const int kernel_rows,
    const int kernel_cols, const int stride) {
  // Unpack and calculate constants.
  const int num_channels = input_volume.size();
  const int input_rows = input_volume.front().rows();
  const int input_cols = input_volume.front().cols();
  const int num_steps_vertical = (input_rows - kernel_rows) / stride + 1;
  const int num_steps_horizontal = (input_cols - kernel_cols) / stride + 1;
  const int num_steps_total = num_steps_horizontal * num_steps_vertical;
  const int num_kernel_elements_per_channel = kernel_cols * kernel_rows;

  // Build input matrix.
  std::vector<Eigen::MatrixXd> input_channels_unrolled;
  for (int j = 0; j < num_channels; ++j) {
    const Eigen::MatrixXd& input_channel = input_volume.at(j);
    Eigen::MatrixXd input_channel_unrolled =
        Eigen::MatrixXd::Zero(num_kernel_elements_per_channel, num_steps_total);
    for (int k = 0; k < num_steps_horizontal; ++k) {
      const int min_ind_col = k * stride;
      for (int l = 0; l < num_steps_vertical; ++l) {
        const int min_ind_row = l * stride;

        // Extract sub-matrix we want to multiply.
        const Eigen::MatrixXd& input_region = input_channel.block(
            min_ind_row, min_ind_col, kernel_rows, kernel_cols);
        const int ind = l + k * num_steps_vertical;  // 0, 1, 2, 3,...
        input_channel_unrolled.col(ind) = Eigen::Map<const Eigen::VectorXd>(
            input_region.data(), input_region.size());
      }
    }
    input_channels_unrolled.emplace_back(std::move(input_channel_unrolled));
  }
  return input_channels_unrolled;
}

std::vector<Eigen::MatrixXd> BuildConvInputMatrixElementWise(
    const std::vector<Eigen::MatrixXd>& input_volume, const int kernel_rows,
    const int kernel_cols, const int stride) {
  // Unpack and calculate constants.
  const int num_channels = input_volume.size();
  const int input_rows = input_volume.front().rows();
  const int input_cols = input_volume.front().cols();
  const int num_steps_vertical = (input_rows - kernel_rows) / stride + 1;
  const int num_steps_horizontal = (input_cols - kernel_cols) / stride + 1;
  const int num_steps_total = num_steps_horizontal * num_steps_vertical;
  const int num_kernel_elements_per_channel = kernel_cols * kernel_rows;

  // Build input matrix.
  std::vector<Eigen::MatrixXd> input_channels_unrolled;
  for (int j = 0; j < num_channels; ++j) {
    const Eigen::MatrixXd& input_channel = input_volume.at(j);
    std::vector<double> input_channel_unrolled(num_kernel_elements_per_channel *
                                               num_steps_total);

    for (int k = 0; k < num_steps_horizontal; ++k) {
      const int min_ind_col = k * stride;
      for (int l = 0; l < num_steps_vertical; ++l) {
        const int min_ind_row = l * stride;
        for (int n = min_ind_col; n < (min_ind_col + kernel_cols); ++n) {
          for (int m = min_ind_row; m < (min_ind_row + kernel_rows); ++m) {
            const int ind = (m - min_ind_row) +
                            (n - min_ind_col) * kernel_rows +
                            l * kernel_cols * kernel_rows +
                            k * kernel_cols * kernel_rows * num_steps_vertical;
            input_channel_unrolled[ind] = input_channel.coeff(m, n);
          }
        }
      }
    }
    input_channels_unrolled.emplace_back(Eigen::Map<Eigen::MatrixXd>(
        input_channel_unrolled.data(), kernel_rows * kernel_cols,
        num_steps_horizontal * num_steps_vertical));
  }
  return input_channels_unrolled;
}

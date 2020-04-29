#include <iostream>

#include "conv.hpp"
#include "conv_tests.hpp"
#include "nn.hpp"

// TODO: Temporarily using a wrapper around Activation, but these
// implementations should be unified somehow.
static InputOutputVolume Activation(
    const InputOutputVolume& input,
    const ActivationFunction& activation_function,
    InputOutputVolume* activation_gradient) {
  const std::vector<double> input_buf = input.GetValues();
  const Eigen::VectorXd& input_vec =
      Eigen::Map<const Eigen::VectorXd>(input_buf.data(), input_buf.size());

  Eigen::MatrixXd grad_mat;
  const Eigen::VectorXd output_vec =
      Activation(input_vec, activation_function, &grad_mat);

  // TODO: Activation() currently returns gradients as a diagonal matrix, but
  // really what we care about is the diagonal vector.
  const Eigen::VectorXd grad_vec =
      grad_mat * Eigen::VectorXd::Ones(grad_mat.cols());

  // Package gradient as InputOutputVolume
  const std::vector<double> grad_buf(grad_vec.data(),
                                     grad_vec.data() + grad_vec.size());
  *activation_gradient = InputOutputVolume(
      grad_buf, input.GetNumChannels(), input.GetNumRows(), input.GetNumCols());

  // Package output as InputOutputVolume
  const std::vector<double> output_buf(output_vec.data(),
                                       output_vec.data() + output_vec.size());
  return InputOutputVolume(output_buf, input.GetNumChannels(),
                           input.GetNumRows(), input.GetNumCols());
}

void TestConvNetGradientsMultiConv() {
  // TODO:
  // - Enable multiple output dimensions (e.g., softmax).
  // - Try handling multiple channels in a single big matrix multiplication to
  // avoid looping over channels.
  // - Generalize to different stride and padding.
  // - Add multiple conv layers (and general specification of layers)
  // - Switch all input/output volumes to 1D buffers that can be reshaped for
  // ease of compatibility between conv and fully connected layers.

  // TODO: Only works for stride of 1.
  const std::size_t stride = 1;

  // Randomly generate input.
  const std::size_t num_channels_input = 1;
  const std::size_t num_rows_input = 3;
  const std::size_t num_cols_input = 3;
  const InputOutputVolume input_volume = GetRandomInputOutputVolume(
      num_channels_input, num_rows_input, num_cols_input);

  // Randomly generate conv layer weights.
  const std::size_t num_kernels = 1;
  const std::size_t num_rows_kernel = 2;
  const std::size_t num_cols_kernel = 2;
  const ConvKernels conv_kernels_1 = GetRandomConvKernels(
      num_kernels, num_channels_input, num_rows_kernel, num_cols_kernel);
  const ConvKernels conv_kernels_2 = GetRandomConvKernels(
      num_kernels, num_kernels, num_rows_kernel, num_cols_kernel);

  // Randomly generate conv biases.
  std::vector<double> conv_biases_1(num_kernels);
  std::vector<double> conv_biases_2(num_kernels);
  std::default_random_engine generator;
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::generate(conv_biases_1.begin(), conv_biases_1.end(),
                [&dist, &generator]() { return dist(generator); });
  std::generate(conv_biases_2.begin(), conv_biases_2.end(),
                [&dist, &generator]() { return dist(generator); });

  // Calculate total number of elements of the conv layer 1 output.
  const size_t num_steps_horizontal_1 =
      (num_cols_input - num_cols_kernel) / stride + 1;
  const size_t num_steps_vertical_1 =
      (num_rows_input - num_rows_kernel) / stride + 1;
  const size_t num_steps_total_1 =
      num_steps_horizontal_1 * num_steps_vertical_1;
  // std::cerr << "Num steps 1: " << num_steps_total_1 << std::endl;

  // Calculate total number of elements of the conv layer 2 output.
  const size_t num_steps_horizontal_2 =
      (num_steps_horizontal_1 - num_cols_kernel) / stride + 1;
  const size_t num_steps_vertical_2 =
      (num_steps_vertical_1 - num_rows_kernel) / stride + 1;
  const size_t num_steps_total_2 =
      num_steps_horizontal_2 * num_steps_vertical_2;
  // std::cerr << "Num steps 2: " << num_steps_total_2 << std::endl;

  // Randomly generate fully connected and output layer weights.
  const std::size_t num_fc = 8;
  const std::size_t num_out = 1;
  const Eigen::MatrixXd W2 =
      Eigen::MatrixXd::Random(num_fc, num_steps_total_2 * num_kernels);
  const Eigen::VectorXd b2 = Eigen::VectorXd::Zero(num_fc);
  const Eigen::MatrixXd W3 = Eigen::MatrixXd::Random(num_out, num_fc);
  const Eigen::VectorXd b3 = Eigen::VectorXd::Zero(num_out);

  // Get nominal output and analytical gradients.
  std::vector<Eigen::MatrixXd> d_output_d_kernel;  // Each element is a channel.
  Eigen::VectorXd d_output_d_bias;
  Eigen::MatrixXd foo;
  const Eigen::VectorXd output = TestConvNetMultiConv(
      input_volume, conv_kernels_1, conv_biases_1, conv_kernels_2,
      conv_biases_2, W2, b2, W3, b3, num_steps_vertical_1,
      num_steps_horizontal_1, num_steps_vertical_2, num_steps_horizontal_2,
      true, &d_output_d_kernel, &d_output_d_bias, &foo);

  // Test numerical gradients of output w.r.t. W_fc.
  const double delta = 1e-6;

  // Numerically compute gradient with respect to inputs
  std::cerr << std::endl;
  const std::vector<double> input_volume_buf = input_volume.GetValues();
  for (int i = 0; i < input_volume_buf.size(); ++i) {
    // Perturb input values.
    std::vector<double> input_volume_buf_perturbed = input_volume_buf;
    input_volume_buf_perturbed.at(i) += delta;
    const InputOutputVolume input_volume_perturbed(
        input_volume_buf_perturbed, input_volume.GetNumChannels(),
        input_volume.GetNumRows(), input_volume.GetNumCols());

    // Evaluate network.
    std::vector<Eigen::MatrixXd> d_output_d_kernel_delta;
    Eigen::VectorXd d_output_d_bias_delta;
    Eigen::MatrixXd foo_delta;
    const Eigen::VectorXd output_delta = TestConvNetMultiConv(
        input_volume_perturbed, conv_kernels_1, conv_biases_1, conv_kernels_2,
        conv_biases_2, W2, b2, W3, b3, num_steps_vertical_1,
        num_steps_horizontal_1, num_steps_vertical_2, num_steps_horizontal_2,
        false, &d_output_d_kernel, &d_output_d_bias, &foo);

    // Compute perturbed output.
    const Eigen::VectorXd numerical_gradient = (output_delta - output) / delta;
    std::cerr << "Input numerical grad " << numerical_gradient << std::endl;
  }

  // Numerically compute gradient with respect to kernel weights 1.
  std::cerr << std::endl;
  const std::vector<double> kernel_weights_1 = conv_kernels_1.GetWeights();
  for (int i = 0; i < kernel_weights_1.size(); ++i) {
    // Perturb kernel weights.
    std::vector<double> kernel_weights_1_perturbed = kernel_weights_1;
    kernel_weights_1_perturbed.at(i) += delta;
    const ConvKernels conv_kernels_1_perturbed(
        kernel_weights_1_perturbed, conv_kernels_1.GetNumKernels(),
        conv_kernels_1.GetNumChannels(), conv_kernels_1.GetNumRows(),
        conv_kernels_1.GetNumCols());

    // Evaluate network.
    std::vector<Eigen::MatrixXd> d_output_d_kernel_delta;
    Eigen::VectorXd d_output_d_bias_delta;
    Eigen::MatrixXd foo_delta;
    const Eigen::VectorXd output_delta = TestConvNetMultiConv(
        input_volume, conv_kernels_1_perturbed, conv_biases_1, conv_kernels_2,
        conv_biases_2, W2, b2, W3, b3, num_steps_vertical_1,
        num_steps_horizontal_1, num_steps_vertical_2, num_steps_horizontal_2,
        false, &d_output_d_kernel, &d_output_d_bias, &foo);

    // Compute perturbed output.
    const Eigen::VectorXd numerical_gradient = (output_delta - output) / delta;
    std::cerr << "Layer 0 numerical grad " << numerical_gradient << std::endl;
  }

  // Numerically compute gradient with respect to kernel weights 2.
  std::cerr << std::endl;
  const std::vector<double> kernel_weights_2 = conv_kernels_2.GetWeights();
  for (int i = 0; i < kernel_weights_2.size(); ++i) {
    // Perturb kernel weights.
    std::vector<double> kernel_weights_2_perturbed = kernel_weights_2;
    kernel_weights_2_perturbed.at(i) += delta;
    const ConvKernels conv_kernels_2_perturbed(
        kernel_weights_2_perturbed, conv_kernels_2.GetNumKernels(),
        conv_kernels_2.GetNumChannels(), conv_kernels_2.GetNumRows(),
        conv_kernels_2.GetNumCols());

    // Evaluate network.
    std::vector<Eigen::MatrixXd> d_output_d_kernel_delta;
    Eigen::VectorXd d_output_d_bias_delta;
    Eigen::MatrixXd foo_delta;
    const Eigen::VectorXd output_delta = TestConvNetMultiConv(
        input_volume, conv_kernels_1, conv_biases_1, conv_kernels_2_perturbed,
        conv_biases_2, W2, b2, W3, b3, num_steps_vertical_1,
        num_steps_horizontal_1, num_steps_vertical_2, num_steps_horizontal_2,
        false, &d_output_d_kernel, &d_output_d_bias, &foo);

    // Compute perturbed output.
    const Eigen::VectorXd numerical_gradient = (output_delta - output) / delta;
    std::cerr << "Layer 1 numerical grad " << numerical_gradient << std::endl;
  }
}

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
    Eigen::VectorXd* d_output_d_bias, Eigen::MatrixXd* foo) {
  const int padding = 0;
  const int stride = 1;

  InputOutputVolume conv_0_output_post_act;
  InputOutputVolume conv_0_output_post_act_grad;
  std::vector<Eigen::MatrixXd> conv_0_input_mat;

  // Conv layer 0
  {
    // Perform convolution.
    std::vector<Eigen::MatrixXd> output_volume_data;
    ConvMatrixMultiplication(
        input_volume.GetVolume(), conv_kernels_0.GetKernels(), conv_biases_0,
        padding, stride, &output_volume_data, &conv_0_input_mat);
    const InputOutputVolume conv_output_volume(output_volume_data);

    // Apply activation.
    conv_0_output_post_act =
        Activation(conv_output_volume, ActivationFunction::SIGMOID,
                   &conv_0_output_post_act_grad);
  }

  InputOutputVolume conv_1_output_post_act;
  InputOutputVolume conv_1_output_post_act_grad;
  std::vector<Eigen::MatrixXd> conv_1_input_mat;

  // Conv layer 1
  {
    // Perform convolution.
    std::vector<Eigen::MatrixXd> output_volume_data;
    ConvMatrixMultiplication(
        conv_0_output_post_act.GetVolume(), conv_kernels_1.GetKernels(),
        conv_biases_1, padding, stride, &output_volume_data, &conv_1_input_mat);
    const InputOutputVolume conv_output_volume(output_volume_data);

    // Apply activation.
    conv_1_output_post_act =
        Activation(conv_output_volume, ActivationFunction::SIGMOID,
                   &conv_1_output_post_act_grad);
  }

  // Convert conv output volume and gradient to vectors to interface with fully
  // connected layer. TODO: Consider how to implement a more abstract layer
  // interface or reshaping layer whose role is just to reshape as appropriate.
  const Eigen::VectorXd conv_1_output_post_act_vec =
      conv_1_output_post_act.GetVolumeVec();
  const Eigen::VectorXd conv_1_output_post_act_grad_vec =
      conv_1_output_post_act_grad.GetVolumeVec();

  // Fully connected layer.
  Eigen::MatrixXd l2_post_act_grad;
  const Eigen::VectorXd l2_pre_act = W2 * conv_1_output_post_act_vec + b2;
  const Eigen::VectorXd l2_post_act =
      Activation(l2_pre_act, ActivationFunction::SIGMOID, &l2_post_act_grad);

  // Output layer.
  Eigen::MatrixXd l3_post_act_grad;
  const Eigen::VectorXd l3_pre_act = W3 * l2_post_act + b3;
  const Eigen::VectorXd l3_post_act =
      Activation(l3_pre_act, ActivationFunction::SIGMOID, &l3_post_act_grad);

  // Compute gradients.
  Eigen::MatrixXd dydw3 = l3_post_act_grad * l2_post_act.transpose();

  Eigen::MatrixXd dydl3 =
      Eigen::MatrixXd::Identity(l3_post_act.size(), l3_post_act.size());

  Eigen::MatrixXd dl3dl2 = l3_post_act_grad * W3;

  // dy/dl2 = dy / dl3  * dl3 / dl2
  Eigen::MatrixXd dydw2 =
      conv_1_output_post_act_vec * l3_post_act_grad * W3 * l2_post_act_grad;
  Eigen::MatrixXd dydl2 = dydl3 * dl3dl2;

  Eigen::MatrixXd dl2dl1 = l2_post_act_grad * W2;

  // TODO: Only works with single channel.
  // Eigen::MatrixXd dydw1 = dydl3 * dl3dl2 * dl2dl1 *
  //                         conv_1_output_post_act_grad *
  //                         conv_1_input_mat.front().transpose();

  Eigen::MatrixXd dydl1 =
      dydl3 * dl3dl2 * dl2dl1 * conv_1_output_post_act_grad_vec.asDiagonal();

  // The values in dydl1 must be backpropagated through the right kernels.
  //
  // TODO: dydl1_cols should be evenly divisible by the number of channels of
  // the l1 output volume (e.g., number of kernels in L1).
  const std::size_t dydl1_cols = dydl1.cols();
  const std::size_t num_kernels_1 = conv_kernels_1.GetNumKernels();
  const std::size_t num_dydl1_per_kernel = dydl1_cols / num_kernels_1;
  assert(dydl1_cols % num_kernels_1 == 0);
  assert(num_dydl1_per_kernel == conv_1_input_mat.front().cols());

  // Wrap values of dydl1 into a matrix where each row corresponds to a kernel.
  // TODO: The wrapping below won't work if dydl1 has multiple rows (e.g., y,
  // the output of the network, is multidimensional).
  assert(dydl1.rows() == 1);
  Eigen::MatrixXd dydl1_wrapped = Eigen::Map<Eigen::MatrixXd>(
      dydl1.data(), num_dydl1_per_kernel, num_kernels_1);

  // Don't forget that this is also a convolution!
  // TODO: See if we can avoid looping over kernels if we can compute
  // convolution with each kernel simultaneously via matrix multiplication,
  // whether the unrolled weight/kernel vector is a matrix of unrolled kernels
  // stacked together.
  std::vector<Eigen::MatrixXd> dydw1_kernels;
  for (std::size_t i = 0; i < num_kernels_1; ++i) {
    dydw1_kernels.emplace_back(conv_1_input_mat.at(i) * dydl1_wrapped);
  }

  if (print) {
    std::cerr << "dydw1" << std::endl;
    for (const auto& dydw : dydw1_kernels) {
      std::cerr << dydw << std::endl;
      std::cerr << std::endl;
    }
  }

  // Don't forget that this is also a convolution!
  // TODO: Only works with single channel.
  // Eigen::MatrixXd dydw1 = dydl1 * conv_1_input_mat.front().transpose();

  // Reshape each column of dydl1_wrapped into a conv kernel.
  std::vector<std::vector<Eigen::MatrixXd>> output_volume_dydl1;
  {
    std::vector<Eigen::MatrixXd> c;
    for (std::size_t i = 0; i < num_kernels_1; ++i) {
      Eigen::VectorXd a = dydl1_wrapped.col(i);
      Eigen::MatrixXd b = Eigen::Map<Eigen::MatrixXd>(
          a.data(), num_steps_vertical_1, num_steps_horizontal_1);
      c.emplace_back(b);
    }

    // TODO: Double check that output_volume_dydl1 should always contain one
    // kernel with a number of channels equal to num_kernels_1.
    output_volume_dydl1.emplace_back(c);
  }

  // Compute dydl0
  //
  // Convolve L1 kernels (input) with dydl1 gradients (kernels)
  //
  // We have N L1 kernels, each with M channels.
  //
  // dydl0 will have depth (num channels) equal to the number of L1 kernels (N).
  std::vector<Eigen::MatrixXd> output_volume_dydl0;
  {
    // For each kernel in conv_kernels_1, perform convolution.
    const std::vector<std::vector<Eigen::MatrixXd>>& ck1 =
        conv_kernels_1.GetKernels();
    const std::size_t num_channels_per_kernel_1 = ck1.front().size();

    // TODO: Either of these loops over channels or kernels may need to reverse
    // (just as we flipped the filters themselves)...or not.
    for (std::size_t j = 0; j < num_channels_per_kernel_1; ++j) {
      std::vector<Eigen::MatrixXd> input_volume_foo;
      for (std::size_t i = 0; i < num_kernels_1; ++i) {
        const Eigen::MatrixXd& ck111 = ck1.at(i).at(j);
        input_volume_foo.emplace_back(
            ck111.rowwise().reverse().colwise().reverse());
      }

      assert(output_volume_dydl1.front().size() == input_volume_foo.size());

      std::vector<Eigen::MatrixXd> input_channels_unrolled_return;
      const std::vector<double> biases{0};

      // TODO: Currently only works with square kernels and inputs due to the
      // padding required for a full convolution.
      // assert(f.rows() == f.cols());
      // assert(dydl1_reshaped.rows() == dydl1_reshaped.cols());

      const std::size_t conv_kernels_rows =
          output_volume_dydl1.front().front().rows();
      const std::size_t conv_kernels_cols =
          output_volume_dydl1.front().front().cols();

      // NOTE: "Full convolution" involves sweeping the filter all the way
      // across, max possible overlap, which can be achieved by doing a
      // "normal" convolution with a padded input.
      std::vector<Eigen::MatrixXd> output_volume_iteration;
      const std::size_t full_conv_padding = conv_kernels_rows - 1;
      ConvMatrixMultiplication(input_volume_foo, output_volume_dydl1, biases,
                               full_conv_padding, 1, &output_volume_iteration,
                               &input_channels_unrolled_return);

      // Is it true that this will always be a single "channel"? This number
      // is the number of "kernels" (dydl1
      assert(output_volume_iteration.size() == 1);
      output_volume_dydl0.emplace_back(output_volume_iteration.front()
                                           .rowwise()
                                           .reverse()
                                           .colwise()
                                           .reverse());
    }
  }

  // Multiply dydl0 with conv_0_output_post_act_grad.
  std::vector<Eigen::MatrixXd> output_volume_dydl0_post_act;
  {
    // TODO: Have previous stage output an InputOutputVolume directly.
    const InputOutputVolume dydl0_iov(output_volume_dydl0);

    // Element-wise product.
    const InputOutputVolume dydl0_iov_post_act =
        conv_0_output_post_act_grad * dydl0_iov;

    // Copy volume to output. TODO: Eliminate unnecesary copy.
    output_volume_dydl0_post_act = dydl0_iov_post_act.GetVolume();
  }

  // Wrap conv_0_output_post_act into a matrix of stacked columns, where
  // each column corresponds to a kernel and multiply it by each element of
  // conv_0_input_mat in a loop.
  const std::size_t num_kernels_0 = conv_kernels_0.GetNumKernels();
  Eigen::MatrixXd dydl0_wrapped = Eigen::MatrixXd::Zero(
      output_volume_dydl0_post_act.front().size(), num_kernels_0);
  for (int i = 0; i < output_volume_dydl0_post_act.size(); ++i) {
    const Eigen::MatrixXd& m = output_volume_dydl0_post_act.at(i);
    dydl0_wrapped.col(i) =
        Eigen::Map<const Eigen::VectorXd>(m.data(), m.size());
  }

  std::vector<Eigen::MatrixXd> dydw0_kernels;
  for (int i = 0; i < conv_0_input_mat.size(); ++i) {
    dydw0_kernels.emplace_back(conv_0_input_mat.at(i) * dydl0_wrapped);
  }
  if (print) {
    std::cerr << "dydw0" << std::endl;
    for (const auto& dydw : dydw0_kernels) {
      std::cerr << dydw << std::endl;
      std::cerr << std::endl;
    }
  }

  // Convert output_volume_dydl0 container to an input/output volume
  std::vector<std::vector<Eigen::MatrixXd>> output_volume_dydl0_expanded{
      output_volume_dydl0_post_act};

  std::vector<Eigen::MatrixXd> output_volume_dydlinput;
  {
    // For each kernel in conv_kernels_1, perform convolution.
    const std::vector<std::vector<Eigen::MatrixXd>>& ck0 =
        conv_kernels_0.GetKernels();
    const std::size_t num_channels_per_kernel_0 = ck0.front().size();

    // TODO: Either of these loops over channels or kernels may need to reverse
    // (just as we flipped the filters themselves)...or not.
    for (std::size_t j = 0; j < num_channels_per_kernel_0; ++j) {
      std::vector<Eigen::MatrixXd> input_volume_foo;
      for (std::size_t i = 0; i < num_kernels_0; ++i) {
        const Eigen::MatrixXd& ck000 = ck0.at(i).at(j);
        input_volume_foo.emplace_back(
            ck000.rowwise().reverse().colwise().reverse());
      }

      assert(output_volume_dydl0_expanded.front().size() ==
             input_volume_foo.size());

      std::vector<Eigen::MatrixXd> input_channels_unrolled_return;
      const std::vector<double> biases{0};

      // TODO: Currently only works with square kernels and inputs due to the
      // padding required for a full convolution.

      const std::size_t conv_kernels_rows =
          output_volume_dydl0_expanded.front().front().rows();
      const std::size_t conv_kernels_cols =
          output_volume_dydl0_expanded.front().front().cols();

      // NOTE: "Full convolution" involves sweeping the filter all the way
      // across, max possible overlap, which can be achieved by doing a
      // "normal" convolution with a padded input.
      std::vector<Eigen::MatrixXd> output_volume_iteration;
      const std::size_t full_conv_padding = conv_kernels_rows - 1;
      ConvMatrixMultiplication(input_volume_foo, output_volume_dydl0_expanded,
                               biases, full_conv_padding, 1,
                               &output_volume_iteration,
                               &input_channels_unrolled_return);

      // Is it true that this will always be a single "channel"? This number
      // is the number of "kernels" (dydl0
      assert(output_volume_iteration.size() == 1);
      output_volume_dydlinput.emplace_back(output_volume_iteration.front()
                                               .rowwise()
                                               .reverse()
                                               .colwise()
                                               .reverse());
    }
  }

  if (print) {
    std::cerr << "dydlinput:" << std::endl;
    for (const auto& d : output_volume_dydlinput) {
      std::cerr << std::endl;
      std::cerr << d << std::endl;
    }
  }

  return l3_post_act;
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
  std::vector<Eigen::MatrixXd> input_channels_unrolled;

  // Compute conv layer.
  Conv(input_volume, conv_kernels, biases, padding, stride, &output_volume);
  ConvMatrixMultiplication(input_volume, conv_kernels, biases, padding, stride,
                           &output_volume_mat_mult, &input_channels_unrolled);

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
  std::vector<Eigen::MatrixXd> input_channels_unrolled;
  ConvMatrixMultiplication(input_volume, conv_kernels, biases, padding, stride,
                           &output_volume, &input_channels_unrolled);
  const std::vector<double> output_values =
      InputOutputVolume(output_volume).GetValues();
  const Eigen::VectorXd output_values_vec = Eigen::Map<const Eigen::VectorXd>(
      output_values.data(), output_values.size());

  // Perturbation magnitude.
  const double delta = 1e-3;

  // Nominal kernel weights.
  const ConvKernels conv_kernels_original(conv_kernels);
  const std::vector<double>& weights = conv_kernels_original.GetWeights();
  const std::size_t num_weights = weights.size();
  const std::size_t num_kernels = conv_kernels_original.GetNumKernels();
  const std::size_t num_channels = conv_kernels_original.GetNumChannels();
  const std::size_t num_rows = conv_kernels_original.GetNumRows();
  const std::size_t num_cols = conv_kernels_original.GetNumCols();

  // Loop over weights.
  for (std::size_t i = 0; i < num_weights; ++i) {
    // Add delta to weights.
    std::vector<double> weights_plus = weights;
    weights_plus.at(i) += delta;

    // Convert back into ConvKernels.
    const ConvKernels conv_kernels_plus(weights_plus, num_kernels, num_channels,
                                        num_rows, num_cols);

    // Evaluate output with perturbed kernel.
    std::vector<Eigen::MatrixXd> output_volume_plus;
    std::vector<Eigen::MatrixXd> input_channels_unrolled_plus;
    ConvMatrixMultiplication(input_volume, conv_kernels_plus.GetKernels(),
                             biases, padding, stride, &output_volume_plus,
                             &input_channels_unrolled_plus);
    const std::vector<double> output_values_plus =
        InputOutputVolume(output_volume_plus).GetValues();
    const Eigen::VectorXd output_values_plus_vec =
        Eigen::Map<const Eigen::VectorXd>(output_values_plus.data(),
                                          output_values_plus.size());

    std::cerr << "original: " << output_values_vec.transpose() << std::endl;
    std::cerr << "plus:     " << output_values_plus_vec.transpose()
              << std::endl;
    std::cerr << "gradient: "
              << (output_values_plus_vec - output_values_vec).transpose() /
                     delta
              << std::endl;

    std::cin.get();
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

void TestFullConv() {
  const int kernel_size = 3;
  const int input_rows = 2;
  const int input_cols = 2;
  const int padding = 2;
  const int stride = 1;
  std::vector<std::vector<Eigen::MatrixXd>> conv_kernels(
      {{Eigen::MatrixXd::Ones(kernel_size, kernel_size)}});
  std::vector<Eigen::MatrixXd> input_volume(
      {Eigen::MatrixXd::Ones(input_rows, input_cols)});
  const std::vector<double> biases{0};
  std::vector<Eigen::MatrixXd> output_volume;
  Conv(input_volume, conv_kernels, biases, padding, stride, &output_volume);

  std::cerr << "Output volume: " << std::endl
            << output_volume.front() << std::endl;
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
  const ConvExample conv_example_3 = GetConvExample3();
  TestConvGradient(conv_example_3);
}

void RunConvKernelTests() {
  const ConvExample conv_example_1 = GetConvExample1();
  TestConvKernels(conv_example_1);
}

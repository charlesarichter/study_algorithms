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
  const std::size_t num_channels_input = 2;
  const std::size_t num_rows_input = 5;
  const std::size_t num_cols_input = 5;
  const InputOutputVolume input_volume = GetRandomInputOutputVolume(
      num_channels_input, num_rows_input, num_cols_input);

  // Randomly generate conv layer weights.
  const std::size_t num_kernels = 3;
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
  const std::size_t num_out = 10;
  const Eigen::MatrixXd W2 =
      Eigen::MatrixXd::Random(num_fc, num_steps_total_2 * num_kernels);
  const Eigen::VectorXd b2 = Eigen::VectorXd::Zero(num_fc);
  const Eigen::MatrixXd W3 = Eigen::MatrixXd::Random(num_out, num_fc);
  const Eigen::VectorXd b3 = Eigen::VectorXd::Zero(num_out);

  // Generate label.
  Eigen::VectorXd label = Eigen::VectorXd::Zero(num_out);
  label(0) = 1;

  // Get nominal output and analytical gradients.
  std::vector<Eigen::MatrixXd> d_output_d_kernel;  // Each element is a channel.
  Eigen::VectorXd d_output_d_bias;
  Eigen::MatrixXd foo;
  const Eigen::VectorXd output = TestConvNetMultiConv(
      input_volume, conv_kernels_1, conv_biases_1, conv_kernels_2,
      conv_biases_2, W2, b2, W3, b3, num_steps_vertical_1,
      num_steps_horizontal_1, num_steps_vertical_2, num_steps_horizontal_2,
      true, label, &d_output_d_kernel, &d_output_d_bias, &foo);

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
        false, label, &d_output_d_kernel, &d_output_d_bias, &foo);

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
        false, label, &d_output_d_kernel, &d_output_d_bias, &foo);

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
        false, label, &d_output_d_kernel, &d_output_d_bias, &foo);

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
    const Eigen::VectorXd& label,
    std::vector<Eigen::MatrixXd>* d_output_d_kernel,
    Eigen::VectorXd* d_output_d_bias, Eigen::MatrixXd* foo) {
  const int padding = 0;
  const int stride = 1;

  const std::size_t num_kernels_1 = conv_kernels_1.GetNumKernels();
  const std::size_t num_kernels_0 = conv_kernels_0.GetNumKernels();

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

  // Evaluate loss.
  Eigen::VectorXd loss_gradient;
  const Eigen::VectorXd loss =
      Loss(l3_post_act, label, LossFunction::CROSS_ENTROPY, &loss_gradient);

  // Network:
  // Pre/Post indicate before/after activation
  //
  //              l0pre   l0post   l1pre   l1post l2pre   l2post l3pre   l3post
  //                |       |        |       |      |       |      |       |
  // Input -> Conv0 -> Act0 -> Conv1 -> Act1 -> Fc2 -> Act2 -> Fc3 -> Act3 -> Y
  //          W0               W1               W2             W3

  // Eigen::MatrixXd dydl3post =
  //     Eigen::MatrixXd::Identity(l3_post_act.size(), l3_post_act.size());

  Eigen::VectorXd dydl3post = loss_gradient;

  Eigen::MatrixXd dl3postdl3pre = l3_post_act_grad;

  Eigen::MatrixXd dydl3pre = (dl3postdl3pre * dydl3post).transpose();

  Eigen::MatrixXd dl3predw3 = l2_post_act;

  Eigen::MatrixXd dydw3 = dl3predw3 * dydl3pre;

  Eigen::MatrixXd dl3predl2post = W3;

  Eigen::MatrixXd dydl2post = dydl3pre * dl3predl2post;

  Eigen::MatrixXd dl2postdl2pre = l2_post_act_grad;

  Eigen::MatrixXd dydl2pre = dydl2post * dl2postdl2pre;

  Eigen::MatrixXd dl2predw2 = conv_1_output_post_act_vec;

  Eigen::MatrixXd dydw2 = dl2predw2 * dydl2pre;

  Eigen::MatrixXd dl2predl1post = W2;

  Eigen::MatrixXd dydl1post = dydl2pre * dl2predl1post;

  Eigen::MatrixXd dl1postdl1pre = conv_1_output_post_act_grad_vec.asDiagonal();

  Eigen::MatrixXd dydl1pre = dydl1post * dl1postdl1pre;

  // The values in dydl1pre must be backpropagated through the right kernels, so
  // we need to reshape the elements correctly to work with them as kernels.
  //
  // Reshape dydl1pre into a single conv kernel.
  std::vector<Eigen::MatrixXd> dydl1pre_volume;
  {
    // TODO: dydl1_cols should be evenly divisible by the number of channels of
    // the l1 output volume (e.g., number of kernels in L1).
    const std::size_t dydl1_cols = dydl1pre.cols();
    const std::size_t num_dydl1_per_kernel = dydl1_cols / num_kernels_1;
    assert(dydl1_cols % num_kernels_1 == 0);
    assert(num_dydl1_per_kernel == conv_1_input_mat.front().cols());

    // Wrap values of dydl1pre into a matrix where each row corresponds to a
    // kernel (or rather a kernel channel).
    //
    // TODO: The wrapping below won't work if dydl1pre has multiple rows (e.g.,
    // y, the output of the network, is multidimensional), but this should not
    // be a problem if we're computing gradients w.r.t. a scalar loss.
    //
    // TODO: This reshaping should already be happening in
    // ConvMatrixMultiplication for calculating dydl0 below.
    assert(dydl1pre.rows() == 1);
    Eigen::MatrixXd dydl1_wrapped = Eigen::Map<Eigen::MatrixXd>(
        dydl1pre.data(), num_dydl1_per_kernel, num_kernels_1);

    // Wrap each column of dydl1_wrapped into a kernel channel shape.
    for (std::size_t i = 0; i < num_kernels_1; ++i) {
      Eigen::VectorXd a = dydl1_wrapped.col(i);
      Eigen::MatrixXd b = Eigen::Map<Eigen::MatrixXd>(
          a.data(), num_steps_vertical_1, num_steps_horizontal_1);
      dydl1pre_volume.emplace_back(b);
    }
  }

  const std::vector<Eigen::MatrixXd> dydw1_kernels =
      ConvWeightGradient(conv_0_output_post_act.GetVolume(), dydl1pre_volume);

  const std::vector<Eigen::MatrixXd> dydl0post_volume =
      ConvGradient(conv_kernels_1, dydl1pre_volume);

  const InputOutputVolume dydl0post_iov(dydl0post_volume);
  const InputOutputVolume dydl0pre_iov =
      conv_0_output_post_act_grad * dydl0post_iov;  // Element-wise product.
  const std::vector<Eigen::MatrixXd> dydl0pre_volume = dydl0pre_iov.GetVolume();

  const std::vector<Eigen::MatrixXd> dydlinput_volume =
      ConvGradient(conv_kernels_0, dydl0pre_volume);

  const std::vector<Eigen::MatrixXd> dydw0_kernels =
      ConvWeightGradient(input_volume.GetVolume(), dydl0pre_volume);

  if (print) {
    std::cerr << "dydw1" << std::endl;
    for (const auto& dydw : dydw1_kernels) {
      std::cerr << dydw << std::endl;
      std::cerr << std::endl;
    }
  }

  if (print) {
    std::cerr << "dydw0" << std::endl;
    for (const auto& dydw : dydw0_kernels) {
      std::cerr << dydw << std::endl;
      std::cerr << std::endl;
    }
  }

  if (print) {
    std::cerr << "dydlinput:" << std::endl;
    for (const auto& d : dydlinput_volume) {
      std::cerr << std::endl;
      std::cerr << d << std::endl;
    }
  }

  return loss;
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

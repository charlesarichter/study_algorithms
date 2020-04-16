#include <iostream>

#include "conv.hpp"
#include "conv_tests.hpp"
#include "nn.hpp"

void TestConvNetGradients() {
  // TODO:
  // - Enable multiple output dimensions (e.g., softmax).
  // - Try handling multiple channels in a single big matrix multiplication to
  // avoid looping over channels.
  // - Generalize to different stride and padding.
  // - Add multiple conv layers (and general specification of layers)
  // - Switch all input/output volumes to 1D buffers that can be reshaped for
  // ease of compatibility between conv and fully connected layers.

  // Randomly generate input.
  const std::size_t num_channels_input = 2;
  const std::size_t num_rows_input = 3;
  const std::size_t num_cols_input = 4;
  const InputOutputVolume input_volume = GetRandomInputOutputVolume(
      num_channels_input, num_rows_input, num_cols_input);

  // Randomly generate conv layer weights.
  const std::size_t num_kernels = 2;
  const std::size_t num_rows_kernel = 2;
  const std::size_t num_cols_kernel = 2;
  const ConvKernels conv_kernels = GetRandomConvKernels(
      num_kernels, num_channels_input, num_rows_kernel, num_cols_kernel);

  // Randomly generate conv biases.
  std::vector<double> conv_biases(num_kernels);
  std::default_random_engine generator;
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::generate(conv_biases.begin(), conv_biases.end(),
                [&dist, &generator]() { return dist(generator); });

  // Calculate total number of elements of the conv layer output.
  const std::size_t stride = 1;  // TODO: Only works for stride of 1.
  const size_t num_steps_horizontal =
      (num_cols_input - num_cols_kernel) / stride + 1;
  const size_t num_steps_vertical =
      (num_rows_input - num_rows_kernel) / stride + 1;
  const size_t num_steps_total = num_steps_horizontal * num_steps_vertical;

  // Randomly generate output layer weights.
  const std::size_t num_outputs = 1;
  const Eigen::MatrixXd W_out =
      Eigen::MatrixXd::Random(num_outputs, num_steps_total * num_kernels);
  const Eigen::VectorXd b_out = Eigen::VectorXd::Zero(num_outputs);

  // Get nominal output and analytical gradients.
  std::vector<Eigen::MatrixXd> d_output_d_kernel;  // Each element is a channel.
  Eigen::VectorXd d_output_d_bias;
  const Eigen::VectorXd output =
      TestConvNet(input_volume, conv_kernels, conv_biases, W_out, b_out,
                  num_steps_total, true, &d_output_d_kernel, &d_output_d_bias);

  // Test numerical gradients of output w.r.t. W_fc.
  const double delta = 1e-6;

  const std::vector<double> kernel_weights = conv_kernels.GetWeights();
  std::vector<Eigen::VectorXd> numerical_gradients(kernel_weights.size());

  // Numerically compute gradient with respect to kernel weights.
  for (int i = 0; i < kernel_weights.size(); ++i) {
    // Perturb kernel weights.
    std::vector<double> kernel_weights_perturbed = kernel_weights;
    kernel_weights_perturbed.at(i) += delta;
    const ConvKernels conv_kernels_perturbed(kernel_weights_perturbed,
                                             num_kernels, num_channels_input,
                                             num_rows_kernel, num_cols_kernel);

    // Evaluate network.
    std::vector<Eigen::MatrixXd> d_output_d_kernel_delta;
    Eigen::VectorXd d_output_d_bias_delta;
    const Eigen::VectorXd output_delta =
        TestConvNet(input_volume, conv_kernels_perturbed, conv_biases, W_out,
                    b_out, num_steps_total, false, &d_output_d_kernel_delta,
                    &d_output_d_bias_delta);

    // Compute perturbed output.
    const Eigen::VectorXd numerical_gradient = (output_delta - output) / delta;
    numerical_gradients.at(i) = numerical_gradient;
    std::cerr << "Numerical gradient " << numerical_gradient << std::endl;
  }

  // Loop through the channels and display gradients corresponding to the kernel
  // weights corresponding to each of them. TODO: Come up with a consistent
  // ordering of weights, either by channel or by kernel, so that the ordering
  // of weights and gradients matches everywhere.
  std::cerr << "Analytical gradient w.r.t. conv weights:" << std::endl;
  for (const Eigen::MatrixXd& d_output_d_kernel_channel : d_output_d_kernel) {
    std::cerr << d_output_d_kernel_channel << std::endl;
  }

  // Numerically compute gradient with respect to conv biases.
  std::vector<Eigen::VectorXd> numerical_bias_gradients(conv_biases.size());
  for (int i = 0; i < conv_biases.size(); ++i) {
    // Perturb kernel weights.
    std::vector<double> conv_biases_perturbed = conv_biases;
    conv_biases_perturbed.at(i) += delta;

    // Evaluate network.
    std::vector<Eigen::MatrixXd> d_output_d_kernel_delta;
    Eigen::VectorXd d_output_d_bias_delta;
    const Eigen::VectorXd output_delta =
        TestConvNet(input_volume, conv_kernels, conv_biases_perturbed, W_out,
                    b_out, num_steps_total, false, &d_output_d_kernel_delta,
                    &d_output_d_bias_delta);

    // Compute perturbed output.
    const Eigen::VectorXd numerical_bias_gradient =
        (output_delta - output) / delta;
    numerical_bias_gradients.at(i) = numerical_bias_gradient;
    std::cerr << "Numerical bias gradient " << numerical_bias_gradient
              << std::endl;
  }

  std::cerr << "Analytical gradient w.r.t. conv biases:" << std::endl;
  std::cerr << d_output_d_bias << std::endl;
}

Eigen::VectorXd TestConvNet(const InputOutputVolume& input_volume,
                            const ConvKernels& conv_kernels,
                            const std::vector<double>& conv_biases,
                            const Eigen::MatrixXd& W_out,
                            const Eigen::VectorXd& b_out,
                            const std::size_t num_steps_total, const bool print,
                            std::vector<Eigen::MatrixXd>* d_output_d_kernel,
                            Eigen::VectorXd* d_output_d_bias) {
  // Compute the first conv layer.
  const int padding = 0;
  const int stride = 1;
  std::vector<Eigen::MatrixXd> output_volume_data;
  std::vector<Eigen::MatrixXd> input_channels_unrolled;
  ConvMatrixMultiplication(input_volume.GetVolume(), conv_kernels.GetKernels(),
                           conv_biases, padding, stride, &output_volume_data,
                           &input_channels_unrolled);
  const InputOutputVolume conv_output_volume(output_volume_data);
  const std::vector<double> conv_output_buf = conv_output_volume.GetValues();
  const Eigen::VectorXd& conv_output = Eigen::Map<const Eigen::VectorXd>(
      conv_output_buf.data(), conv_output_buf.size());

  // std::cerr << "output volume data size: " << output_volume_data.size() <<
  // std::endl;

  // Apply activation function to conv output.
  // TODO: Enable Activation() to more easily operate on 1D data buffers and
  // then make InputOutputVolume store the data as a 1D buffer that can be
  // passed in, rather than having to reshape/map the data.
  Eigen::MatrixXd conv_output_post_act_grad;
  const Eigen::VectorXd conv_output_post_act = Activation(
      conv_output, ActivationFunction::SIGMOID, &conv_output_post_act_grad);

  // Output layer.
  Eigen::MatrixXd output_post_act_grad;
  const Eigen::VectorXd output_pre_act = W_out * conv_output_post_act + b_out;
  const Eigen::VectorXd output_post_act = Activation(
      output_pre_act, ActivationFunction::SIGMOID, &output_post_act_grad);

  // if (print) {
  // std::cerr << "Output: " << output_post_act.transpose() << std::endl;
  // std::cerr << "output_post_act_grad: " << std::endl
  //           << output_post_act_grad << std::endl;
  // std::cerr << "W_out: " << std::endl << W_out << std::endl;
  // std::cerr << "conv_output_post_act_grad: " << std::endl
  //           << conv_output_post_act_grad << std::endl;
  // std::cerr << "input_channels_unrolled: " << std::endl
  //           << input_channels_unrolled.front() << std::endl;
  //
  // assert(input_channels_unrolled.size() == 1);
  // const Eigen::MatrixXd d_activated_conv_layer_d_kernel_weights =
  //     conv_output_post_act_grad *
  //     input_channels_unrolled.front().transpose();
  // std::cerr << "d_activated_conv_layer_d_kernel_weights: " << std::endl
  //           << d_activated_conv_layer_d_kernel_weights << std::endl;
  // }

  // Compute gradient of output w.r.t. kernel weights:
  //
  // output_post_act_grad * W_out * conv_output_post_act_grad *
  // unrolled_input_matrix;
  const Eigen::MatrixXd a =
      output_post_act_grad * W_out * conv_output_post_act_grad;
  const Eigen::MatrixXd b = Eigen::Map<const Eigen::MatrixXd>(
      a.data(), num_steps_total, conv_kernels.GetNumKernels());

  // Loop through the channels.
  for (const Eigen::MatrixXd& input_channel_unrolled :
       input_channels_unrolled) {
    const Eigen::MatrixXd dydw =
        b.transpose() * input_channel_unrolled.transpose();
    d_output_d_kernel->emplace_back(dydw);
  }

  // Because we use the same bias value for each step of the convolution (per
  // kernel), we need to essentially sum the gradient values originating from
  // each step that pertain to the same bias value. This operation is the same
  // as summing the columns of b.
  *d_output_d_bias = Eigen::VectorXd::Ones(b.rows()).transpose() * b;

  return output_post_act;
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

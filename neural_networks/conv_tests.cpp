#include <iostream>

#include "conv.hpp"
#include "conv_tests.hpp"
#include "nn.hpp"

void TestConvNetGradients() {
  // Randomly generate input.
  const std::size_t num_channels_input = 3;
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
  const std::vector<double> biases(num_kernels, 0);

  // Randomly generate fully connected layer weights.
  // TODO: Compute num_fully_connected_inputs based on input and conv dims.
  const std::size_t num_fully_connected_inputs = 12;
  const std::size_t num_fully_connected_outputs = 16;
  const Eigen::MatrixXd W_fc = Eigen::MatrixXd::Random(
      num_fully_connected_outputs, num_fully_connected_inputs);
  const Eigen::VectorXd b_fc =
      Eigen::VectorXd::Random(num_fully_connected_outputs);

  // Randomly generate output layer weights.
  const std::size_t num_outputs = 2;
  const Eigen::MatrixXd W_out =
      Eigen::MatrixXd::Random(num_outputs, num_fully_connected_outputs);
  const Eigen::VectorXd b_out = Eigen::VectorXd::Random(num_outputs);

  // Gradient containers.
  std::vector<std::vector<Eigen::MatrixXd>> manual_weight_gradients;
  std::vector<std::vector<Eigen::VectorXd>> manual_bias_gradients;

  const Eigen::VectorXd output =
      TestConvNet(input_volume, conv_kernels, W_fc, b_fc, W_out, b_out,
                  &manual_weight_gradients, &manual_bias_gradients);

  // Test numerical gradients of output w.r.t. W_fc.
  const double delta = 1e-6;
  std::vector<std::vector<Eigen::VectorXd>> numerical_gradient(
      W_fc.rows(), std::vector<Eigen::VectorXd>(W_fc.cols()));

  // Numerically compute gradient.
  for (int i = 0; i < W_fc.rows(); ++i) {
    for (int j = 0; j < W_fc.cols(); ++j) {
      // Perturbed weights.
      Eigen::MatrixXd W_fc_delta = W_fc;
      W_fc_delta(i, j) += delta;

      const Eigen::VectorXd output_delta =
          TestConvNet(input_volume, conv_kernels, W_fc_delta, b_fc, W_out,
                      b_out, &manual_weight_gradients, &manual_bias_gradients);

      // Compute perturbed output.
      numerical_gradient.at(i).at(j) = (output_delta - output) / delta;
    }
  }

  // Print numerical and analytical gradients of the first output element with
  // respect to the fully connected layer weight element at position (0, 0).
  const Eigen::VectorXd& num_grad_00 = numerical_gradient.at(0).at(0);
  const double analytical_grad_0_00 =
      manual_weight_gradients.at(2).at(0).coeff(0, 0);
  const double analytical_grad_1_00 =
      manual_weight_gradients.at(2).at(1).coeff(0, 0);
  std::cerr << "Numerical grad: " << num_grad_00.transpose() << std::endl;
  std::cerr << "Analytical grad: " << analytical_grad_0_00 << " "
            << analytical_grad_1_00 << std::endl;
}

Eigen::VectorXd TestConvNet(
    const InputOutputVolume& input_volume, const ConvKernels& conv_kernels,
    const Eigen::MatrixXd& W_fc, const Eigen::VectorXd& b_fc,
    const Eigen::MatrixXd& W_out, const Eigen::VectorXd& b_out,
    std::vector<std::vector<Eigen::MatrixXd>>* manual_weight_gradients,
    std::vector<std::vector<Eigen::VectorXd>>* manual_bias_gradients) {
  // Input -> Conv layer 1 -> activation -> Conv layer 2 -> FC -> activation

  // TODO: Actual biases.
  const std::vector<double> biases(conv_kernels.GetKernels().size(), 0);

  // Compute the first conv layer.
  const int padding = 0;
  const int stride = 1;
  std::vector<Eigen::MatrixXd> output_volume_data;
  ConvMatrixMultiplication(input_volume.GetVolume(), conv_kernels.GetKernels(),
                           biases, padding, stride, &output_volume_data);
  const InputOutputVolume first_conv_layer_output(output_volume_data);

  // Unroll output.
  const std::vector<double>& first_conv_layer_output_values =
      first_conv_layer_output.GetValues();
  const Eigen::VectorXd& layer_1_pre_act =
      Eigen::Map<const Eigen::VectorXd>(first_conv_layer_output_values.data(),
                                        first_conv_layer_output_values.size());

  // Apply activation function to conv output.
  // TODO: Enable Activation() to more easily operate on 1D data buffers and
  // then make InputOutputVolume store the data as a 1D buffer that can be
  // passed in, rather than having to reshape/map the data.
  Eigen::MatrixXd first_conv_layer_output_activated_gradient;
  const Eigen::VectorXd l1_post_act =
      Activation(layer_1_pre_act, ActivationFunction::RELU,
                 &first_conv_layer_output_activated_gradient);

  // Fully connected layer.
  Eigen::MatrixXd l2_post_act_grad;
  const Eigen::VectorXd l2_pre_act = W_fc * l1_post_act + b_fc;
  const Eigen::VectorXd l2_post_act =
      Activation(l2_pre_act, ActivationFunction::RELU, &l2_post_act_grad);

  // Output layer.
  Eigen::MatrixXd l3_post_act_grad;
  const Eigen::VectorXd l3_pre_act = W_out * l2_post_act + b_out;
  const Eigen::VectorXd l3_post_act =
      Activation(l3_pre_act, ActivationFunction::SOFTMAX, &l3_post_act_grad);

  // std::cerr << "Output: " << l3_post_act.transpose() << std::endl;

  /////////////////////////////////////////////////////////////////////////////

  const int num_layers = 4;
  const int output_dimension = W_out.rows();

  manual_weight_gradients->resize(num_layers);
  manual_bias_gradients->resize(num_layers);
  for (int i = 0; i < num_layers; ++i) {
    manual_weight_gradients->at(i).resize(output_dimension);
    manual_bias_gradients->at(i).resize(output_dimension);
  }

  // Loop over the network outputs. Essentially we are structuring this
  // computation as if we have a separate network for each of the outputs. Use
  // this to find reused computation and condense/optimize if we can.
  for (int j = 0; j < output_dimension; ++j) {
    // dy/dA3 = f3'(layer_2_pre_act) * layer_2_post_act
    const Eigen::MatrixXd a = Eigen::MatrixXd::Ones(1, 1);
    const Eigen::MatrixXd dydw3 =
        (a * l3_post_act_grad.row(j)).transpose() * l2_post_act.transpose();

    // dy/db3 = f3'(layer_2_pre_act)
    const Eigen::VectorXd dydb3 = (a * l3_post_act_grad.row(j)).transpose();
    // std::cerr << "dydb3 " << std::endl << dydb3 << std::endl;

    manual_weight_gradients->at(3).at(j) = dydw3;
    manual_bias_gradients->at(3).at(j) = dydb3;

    // dy/dA2 = f3'(layer_3_pre_act) * A3
    //         * f2'(layer_2_pre_act) * layer_1_post_act
    const Eigen::MatrixXd b =
        a * l3_post_act_grad.row(j) * W_out;  // nn.weights.at(3);
    const Eigen::MatrixXd dydw2 =
        (b * l2_post_act_grad).transpose() * l1_post_act.transpose();
    // std::cerr << "dydw2 " << std::endl << dydw2 << std::endl;

    // dy/db2 = f3'(layer_3_pre_act) * A3
    //         * f2'(layer_2_pre_act)
    const Eigen::VectorXd dydb2 = (b * l2_post_act_grad).transpose();
    // std::cerr << "dydb2 " << std::endl << dydb2 << std::endl;

    manual_weight_gradients->at(2).at(j) = dydw2;
    manual_bias_gradients->at(2).at(j) = dydb2;
    //
    // // dy/dA1 = f3'(layer_3_pre_act) * A3
    // //        * f2'(layer_2_pre_act) * A2
    // //        * f1'(layer_1_pre_act) * layer_0_post_act
    // const Eigen::MatrixXd c = b * l2_post_act_grad * nn.weights.at(2);
    // const Eigen::MatrixXd dydw1 =
    //     (c * l1_post_act_grad).transpose() * l0_post_act.transpose();
    // // std::cerr << "dydw1 " << std::endl << dydw1 << std::endl;
    //
    // // dy/db1 = f3'(layer_3_pre_act) * A3
    // //        * f2'(layer_2_pre_act) * A2
    // //        * f1'(layer_1_pre_act)
    // const Eigen::MatrixXd dydb1 = (c * l1_post_act_grad).transpose();
    // // std::cerr << "dydb1 " << std::endl << dydb1 << std::endl;
    //
    // manual_weight_gradients->at(1).at(j) = dydw1;
    // manual_bias_gradients->at(1).at(j) = dydb1;
    //
    // // dy/dA0 = f3'(layer_3_pre_act) * A3
    // //        * f2'(layer_2_pre_act) * A2
    // //        * f1'(layer_1_pre_act) * A1
    // //        * f0'(layer_0_pre_act) * input
    // const Eigen::MatrixXd d = c * l1_post_act_grad * nn.weights.at(1);
    // const Eigen::MatrixXd dydw0 =
    //     (d * l0_post_act_grad).transpose() * input.transpose();
    // // std::cerr << "dydw0 " << std::endl << dydw0 << std::endl;
    //
    // // dy/db0 = f3'(layer_3_pre_act) * A3
    // //        * f2'(layer_2_pre_act) * A2
    // //        * f1'(layer_1_pre_act) * A1
    // //        * f0'(layer_0_pre_act)
    // const Eigen::MatrixXd dydb0 = (d * l0_post_act_grad).transpose();
    // // std::cerr << "dydb0 " << std::endl << dydb0 << std::endl;
    //
    // manual_weight_gradients->at(0).at(j) = dydw0;
    // manual_bias_gradients->at(0).at(j) = dydb0;
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
    ConvMatrixMultiplication(input_volume, conv_kernels_plus.GetKernels(),
                             biases, padding, stride, &output_volume_plus);
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

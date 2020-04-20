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

  // const ConvKernels conv_kernels_2 = GetRandomConvKernels(
  //     num_kernels, num_kernels, num_rows_kernel, num_cols_kernel);

  // Randomly generate conv biases.
  std::vector<double> conv_biases_1(num_kernels);
  // std::vector<double> conv_biases_2(num_kernels);
  std::default_random_engine generator;
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::generate(conv_biases_1.begin(), conv_biases_1.end(),
                [&dist, &generator]() { return dist(generator); });
  // std::generate(conv_biases_2.begin(), conv_biases_2.end(),
  //               [&dist, &generator]() { return dist(generator); });

  // Calculate total number of elements of the conv layer 1 output.
  const size_t num_steps_horizontal_1 =
      (num_cols_input - num_cols_kernel) / stride + 1;
  const size_t num_steps_vertical_1 =
      (num_rows_input - num_rows_kernel) / stride + 1;
  const size_t num_steps_total_1 =
      num_steps_horizontal_1 * num_steps_vertical_1;
  std::cerr << "Num steps 1: " << num_steps_total_1 << std::endl;

  // // Calculate total number of elements of the conv layer 2 output.
  // const size_t num_steps_horizontal_2 =
  //     (num_steps_horizontal_1 - num_cols_kernel) / stride + 1;
  // const size_t num_steps_vertical_2 =
  //     (num_steps_vertical_1 - num_rows_kernel) / stride + 1;
  // const size_t num_steps_total_2 =
  //     num_steps_horizontal_2 * num_steps_vertical_2;
  // // std::cerr << "Num steps 2: " << num_steps_total_2 << std::endl;

  // Randomly generate fully connected and output layer weights.
  const std::size_t num_fc = 8;
  const std::size_t num_out = 1;
  const Eigen::MatrixXd W2 =
      Eigen::MatrixXd::Random(num_fc, num_steps_total_1 * num_kernels);
  const Eigen::VectorXd b2 = Eigen::VectorXd::Zero(num_fc);
  const Eigen::MatrixXd W3 = Eigen::MatrixXd::Random(num_out, num_fc);
  const Eigen::VectorXd b3 = Eigen::VectorXd::Zero(num_out);

  // Get nominal output and analytical gradients.
  std::vector<Eigen::MatrixXd> d_output_d_kernel;  // Each element is a channel.
  Eigen::VectorXd d_output_d_bias;
  Eigen::MatrixXd foo;
  const Eigen::VectorXd output =
      TestConvNetMultiConv(input_volume, conv_kernels_1, conv_biases_1, W2, b2,
                           W3, b3, num_steps_vertical_1, num_steps_horizontal_1,
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
        input_volume_perturbed, conv_kernels_1, conv_biases_1, W2, b2, W3, b3,
        num_steps_vertical_1, num_steps_horizontal_1, false, &d_output_d_kernel,
        &d_output_d_bias, &foo);

    // Compute perturbed output.
    const Eigen::VectorXd numerical_gradient = (output_delta - output) / delta;
    std::cerr << "Layer 0 numerical grad " << numerical_gradient << std::endl;
  }

  // Numerically compute gradient with respect to kernel weights 1.
  std::cerr << std::endl;
  const std::vector<double> kernel_weights_1 = conv_kernels_1.GetWeights();
  for (int i = 0; i < kernel_weights_1.size(); ++i) {
    // Perturb kernel weights.
    std::vector<double> kernel_weights_1_perturbed = kernel_weights_1;
    kernel_weights_1_perturbed.at(i) += delta;
    const ConvKernels conv_kernels_1_perturbed(
        kernel_weights_1_perturbed, num_kernels, num_kernels, num_rows_kernel,
        num_cols_kernel);

    // Evaluate network.
    std::vector<Eigen::MatrixXd> d_output_d_kernel_delta;
    Eigen::VectorXd d_output_d_bias_delta;
    Eigen::MatrixXd foo_delta;
    const Eigen::VectorXd output_delta = TestConvNetMultiConv(
        input_volume, conv_kernels_1_perturbed, conv_biases_1, W2, b2, W3, b3,
        num_steps_vertical_1, num_steps_horizontal_1, false, &d_output_d_kernel,
        &d_output_d_bias, &foo);

    // Compute perturbed output.
    const Eigen::VectorXd numerical_gradient = (output_delta - output) / delta;
    std::cerr << "Layer 0 numerical grad " << numerical_gradient << std::endl;
  }

  // // Numerically compute gradient with respect to kernel weights 2.
  // std::cerr << std::endl;
  // const std::vector<double> kernel_weights_2 = conv_kernels_2.GetWeights();
  // for (int i = 0; i < kernel_weights_2.size(); ++i) {
  //   // Perturb kernel weights.
  //   std::vector<double> kernel_weights_2_perturbed = kernel_weights_2;
  //   kernel_weights_2_perturbed.at(i) += delta;
  //   const ConvKernels conv_kernels_2_perturbed(
  //       kernel_weights_2_perturbed, num_kernels, num_kernels,
  //       num_rows_kernel, num_cols_kernel);
  //
  //   // Evaluate network.
  //   std::vector<Eigen::MatrixXd> d_output_d_kernel_delta;
  //   Eigen::VectorXd d_output_d_bias_delta;
  //   Eigen::MatrixXd foo_delta;
  //   const Eigen::VectorXd output_delta = TestConvNetMultiConv(
  //       input_volume, conv_kernels_1, conv_biases_1,
  //       conv_kernels_2_perturbed, conv_biases_2, W2, b2, W3, b3,
  //       num_steps_total_1, num_steps_total_2, false,
  //       &d_output_d_kernel_delta, &d_output_d_bias_delta, &foo_delta);
  //
  //   // Compute perturbed output.
  //   const Eigen::VectorXd numerical_gradient = (output_delta - output) /
  //   delta;
  //   // std::cerr << "Layer 1 numerical grad " << numerical_gradient <<
  //   // std::endl;
  // }
  //
  // // Numerically compute gradient with respect to FC weights.
  // std::cerr << std::endl;
  // Eigen::MatrixXd num_grad_mat = Eigen::MatrixXd::Zero(W2.rows(), W2.cols());
  // for (int i = 0; i < W2.rows(); ++i) {
  //   for (int j = 0; j < W2.cols(); ++j) {
  //     // Perturb weights.
  //     Eigen::MatrixXd W2_perturbed = W2;
  //     W2_perturbed(i, j) += delta;
  //
  //     // Evaluate network.
  //     std::vector<Eigen::MatrixXd> d_output_d_kernel_delta;
  //     Eigen::VectorXd d_output_d_bias_delta;
  //     Eigen::MatrixXd foo_delta;
  //     const Eigen::VectorXd output_delta = TestConvNetMultiConv(
  //         input_volume, conv_kernels_1, conv_biases_1, conv_kernels_2,
  //         conv_biases_2, W2_perturbed, b2, W3, b3, num_steps_total_1,
  //         num_steps_total_2, false, &d_output_d_kernel_delta,
  //         &d_output_d_bias_delta, &foo_delta);
  //
  //     // Compute perturbed output.
  //     const Eigen::VectorXd numerical_gradient =
  //         (output_delta - output) / delta;
  //     // std::cerr << "Layer 1 numerical grad " << numerical_gradient <<
  //     // std::endl;
  //     num_grad_mat(i, j) = numerical_gradient(0);
  //   }
  // }
  // // std::cerr << "Layer 2 numerical grad: " << std::endl
  // //           << num_grad_mat.transpose() << std::endl;
}

Eigen::VectorXd TestConvNetMultiConv(
    const InputOutputVolume& input_volume, const ConvKernels& conv_kernels_0,
    const std::vector<double>& conv_biases_0, const Eigen::MatrixXd& W2,
    const Eigen::VectorXd& b2, const Eigen::MatrixXd& W3,
    const Eigen::VectorXd& b3, const std::size_t num_steps_vertical_0,
    const std::size_t num_steps_horizontal_0, const bool print,
    std::vector<Eigen::MatrixXd>* d_output_d_kernel,
    Eigen::VectorXd* d_output_d_bias, Eigen::MatrixXd* foo) {
  const int padding = 0;
  const int stride = 1;

  Eigen::VectorXd conv_0_output_post_act;
  Eigen::MatrixXd conv_0_output_post_act_grad;
  std::vector<Eigen::MatrixXd> conv_0_input_mat;
  int conv_output_rows = 0;
  int conv_output_cols = 0;

  // Conv layer 0
  {
    std::vector<Eigen::MatrixXd> output_volume_data;
    ConvMatrixMultiplication(
        input_volume.GetVolume(), conv_kernels_0.GetKernels(), conv_biases_0,
        padding, stride, &output_volume_data, &conv_0_input_mat);
    const InputOutputVolume conv_output_volume(output_volume_data);
    const std::vector<double> conv_output_buf = conv_output_volume.GetValues();
    const Eigen::VectorXd& conv_output = Eigen::Map<const Eigen::VectorXd>(
        conv_output_buf.data(), conv_output_buf.size());
    conv_0_output_post_act = Activation(
        conv_output, ActivationFunction::SIGMOID, &conv_0_output_post_act_grad);
    // const std::vector<double> conv_0_output_post_act_buf(
    //     conv_0_output_post_act_vec.data(),
    //     conv_0_output_post_act_vec.data() +
    //     conv_0_output_post_act_vec.size());
    // conv_0_output_post_act = InputOutputVolume(
    //     conv_0_output_post_act_buf, conv_output_volume.GetNumChannels(),
    //     conv_output_volume.GetNumRows(), conv_output_volume.GetNumCols());

    conv_output_rows = output_volume_data.front().rows();
    conv_output_cols = output_volume_data.front().cols();
    // if (print) {
    //   std::cerr << "conv outputs: " << conv_output_rows << ", "
    //             << conv_output_cols << std::endl;
    // }
  }

  // Eigen::VectorXd conv_1_output_post_act;
  // Eigen::MatrixXd conv_1_output_post_act_grad;
  // std::vector<Eigen::MatrixXd> conv_1_input_mat;
  //
  // // Conv layer 1
  // {
  //   std::vector<Eigen::MatrixXd> output_volume_data;
  //   ConvMatrixMultiplication(
  //       input_volume.GetVolume(), conv_kernels_1.GetKernels(), conv_biases_1,
  //       padding, stride, &output_volume_data, &conv_1_input_mat);
  //   const InputOutputVolume conv_output_volume(output_volume_data);
  //   const std::vector<double> conv_output_buf =
  //   conv_output_volume.GetValues(); const Eigen::VectorXd& conv_output =
  //   Eigen::Map<const Eigen::VectorXd>(
  //       conv_output_buf.data(), conv_output_buf.size());
  //   conv_1_output_post_act = Activation(
  //       conv_output, ActivationFunction::SIGMOID,
  //       &conv_1_output_post_act_grad);
  //
  //   // std::cerr << "Layer 1 conv output " << conv_output_volume.GetNumRows()
  //   //           << " " << conv_output_volume.GetNumCols() << std::endl;
  // }

  // Fully connected layer.
  Eigen::MatrixXd l2_post_act_grad;
  const Eigen::VectorXd l2_pre_act = W2 * conv_0_output_post_act + b2;
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
      conv_0_output_post_act * l3_post_act_grad * W3 * l2_post_act_grad;
  Eigen::MatrixXd dydl2 = dydl3 * dl3dl2;

  Eigen::MatrixXd dl2dl1 = l2_post_act_grad * W2;

  // TODO: Only works with single channel.
  Eigen::MatrixXd dydw1 = dydl3 * dl3dl2 * dl2dl1 *
                          conv_0_output_post_act_grad *
                          conv_0_input_mat.front().transpose();

  if (print) {
    std::cerr << std::endl;
    std::cerr << dydw1 << std::endl;
    std::cerr << std::endl;
  }

  Eigen::MatrixXd dydl1 = dydl3 * dl3dl2 * dl2dl1 * conv_0_output_post_act_grad;

  // Compute dydl0
  // Shape of l1 output is a conv output volume.
  Eigen::MatrixXd dydl1_reshaped = Eigen::Map<Eigen::MatrixXd>(
      dydl1.data(), num_steps_vertical_0, num_steps_horizontal_0);
  // if (print) {
  //   std::cerr << dydl1 << std::endl;
  //   std::cerr << std::endl;
  //   std::cerr << dydl1_reshaped << std::endl;
  //   std::cerr << std::endl;
  // }

  std::vector<Eigen::MatrixXd> output_volume;
  {
    // "Full convolution" between dydl1_reshaped and conv_kernels_1 (flipped?).
    Eigen::MatrixXd f = conv_kernels_0.GetKernels()
                            .front()
                            .front()
                            .rowwise()
                            .reverse()
                            .colwise()
                            .reverse();
    const std::vector<Eigen::MatrixXd> input_volume{f};
    const std::vector<std::vector<Eigen::MatrixXd>> conv_kernels{
        {dydl1_reshaped}};
    std::vector<Eigen::MatrixXd> input_channels_unrolled_return;
    const std::vector<double> biases{0};

    // TODO: Currently only works with square kernels and inputs due to the
    // padding required for a full convolution.
    assert(f.rows() == f.cols());
    assert(dydl1_reshaped.rows() == dydl1_reshaped.cols());

    const std::size_t input_volume_rows = f.rows();
    const std::size_t input_volume_cols = f.cols();
    const std::size_t conv_kernels_rows = dydl1_reshaped.rows();
    const std::size_t conv_kernels_cols = dydl1_reshaped.cols();
    // if (print) {
    //   std::cerr << "Input volume: " << input_volume_rows << " x "
    //             << input_volume_cols << std::endl;
    //   std::cerr << "Conv kernels: " << conv_kernels_rows << " x "
    //             << conv_kernels_cols << std::endl;
    // }

    // NOTE: "Full convolution" involves sweeping the filter all the way across,
    // max possible overlap, which can be achieved by doing a "normal"
    // convolution with a padded input.
    const std::size_t full_conv_padding = conv_kernels_rows - 1;

    // TODO: Symmetrical padding may not work for non-square kernels
    ConvMatrixMultiplication(input_volume, conv_kernels, biases,
                             full_conv_padding, 1, &output_volume,
                             &input_channels_unrolled_return);

    // Conv(input_volume, conv_kernels, biases, padding, 1, &output_volume);
    // std::cerr << "Output volume: " << std::endl
    //           << output_volume.front() << std::endl;
    // std::cerr << "Input unrolled: " << std::endl
    //           << input_channels_unrolled_return.front().rows() << " "
    //           << input_channels_unrolled_return.front().cols() << std::endl;
    // }
  }
  const Eigen::MatrixXd dydl0 = output_volume.front();
  if (print) {
    std::cerr << dydl0 << std::endl;
  }

  // Eigen::MatrixXd dydl0 = 0;

  // const std::vector<double> W1_buf = conv_kernels_1.GetWeights();
  // const Eigen::VectorXd W1 =
  //     Eigen::Map<const Eigen::VectorXd>(W1_buf.data(), W1_buf.size());
  //
  // const Eigen::MatrixXd A = l3_post_act_grad * W3 * l2_post_act_grad * W2 *
  //                           conv_1_output_post_act_grad;

  // if (print) {
  //   std::cerr << A << std::endl << std::endl;
  //   std::cerr << W1 << std::endl << std::endl;
  //   std::cerr << conv_0_output_post_act_grad << std::endl << std::endl;
  //   std::cerr << conv_0_input_mat.front() << std::endl << std::endl;
  //
  //   std::cerr << conv_0_output_post_act_grad *
  //                    conv_0_input_mat.front().transpose()
  //             << std::endl
  //             << std::endl;
  // }

  // if (print) {
  //   std::cerr << std::endl;
  //   std::cerr << "dydw3: " << std::endl << dydw3 << std::endl;
  //   std::cerr << std::endl;
  //   std::cerr << "dydw2: " << std::endl << dydw2 << std::endl;
  //   std::cerr << std::endl;
  //   std::cerr << "dydw1: " << std::endl << dydw1 << std::endl;
  // }

  // Compute gradient of output w.r.t. kernel weights:
  //
  // output_post_act_grad * W_out * conv_output_post_act_grad *
  // unrolled_input_matrix;
  // const Eigen::MatrixXd a =
  //     output_post_act_grad * W_out * conv_2_output_post_act_grad;
  //
  // const Eigen::MatrixXd b = Eigen::Map<const Eigen::MatrixXd>(
  //     a.data(), num_steps_total_2, conv_kernels_2.GetNumKernels());

  // // Loop through the channels.
  // for (const Eigen::MatrixXd& input_channel_unrolled : conv_2_input_mat) {
  //   const Eigen::MatrixXd dydw =
  //       b.transpose() * input_channel_unrolled.transpose();
  //   d_output_d_kernel->emplace_back(dydw);
  // }
  // const Eigen::MatrixXd c =
  //     b.transpose() * conv_2_input_mat.front().transpose();
  //
  // const Eigen::MatrixXd d = Eigen::Map<const Eigen::MatrixXd>(
  //     c.data(), num_steps_total_1, conv_kernels_1.GetNumKernels());
  //
  // const Eigen::MatrixXd grad =
  //     d.transpose() * conv_1_input_mat.front().transpose();
  //
  // if (print) {
  //   std::cerr << "Gradient: " << std::endl << grad << std::endl;
  //   std::cerr << "foo: " << conv_1_output_post_act_grad << std::endl;
  // }

  // Because we use the same bias value for each step of the convolution (per
  // kernel), we need to essentially sum the gradient values originating from
  // each step that pertain to the same bias value. This operation is the same
  // as summing the columns of b.
  // *d_output_d_bias = Eigen::VectorXd::Ones(b.rows()).transpose() * b;

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

#include "layer.hpp"

#include <iostream>
#include <random>

#include "conv.hpp"

int LayerFC::GetNumParameters() const {
  return num_inputs_ * num_outputs_ + num_outputs_;
}

std::vector<double> LayerFC::GetRandomParameters() const {
  const double weight_coefficient = GetWeightCoefficient(activation_function_);

  // Generate weights.
  const int num_weights = num_inputs_ * num_outputs_;
  const std::vector<double> weights =
      GetRandomVector(num_weights, -1 * weight_coefficient, weight_coefficient);

  // Generate biases.
  const std::vector<double> biases = GetRandomVector(
      num_outputs_, -1 * weight_coefficient, weight_coefficient);

  // Output.
  std::vector<double> parameters;
  parameters.insert(parameters.end(), weights.begin(), weights.end());
  parameters.insert(parameters.end(), biases.begin(), biases.end());
  return parameters;
}

void LayerFC::ForwardPass(const std::vector<double>& input,
                          const std::vector<double>& parameters,
                          std::vector<double>* output,
                          std::vector<double>* activation_gradient) const {
  const std::size_t num_weights = num_inputs_ * num_outputs_;
  const std::size_t num_biases = num_outputs_;

  assert(input.size() == num_inputs_);
  assert(parameters.size() == GetNumParameters());

  const Eigen::Map<const Eigen::VectorXd> input_vec(input.data(), input.size());

  // Reshape parameters into weight matrix and bias vector.
  // NOTE: `parameters` contains both weights and biases. The Eigen::Map
  // `W` should just take the first num_outputs_ * num_inputs_ values.
  const Eigen::Map<const Eigen::MatrixXd> W(parameters.data(), num_outputs_,
                                            num_inputs_);
  const Eigen::Map<const Eigen::VectorXd> b(parameters.data() + num_weights,
                                            num_biases);

  // Compute pre-activation result.
  const Eigen::VectorXd pre_activation = W * input_vec + b;

  // Compute post-activation result and gradient.
  // TODO: Consider implementing Activation function to operate directly on
  // std::vector<double> input.
  Eigen::MatrixXd activation_gradient_mat;
  const Eigen::VectorXd post_activation = Activation(
      pre_activation, activation_function_, &activation_gradient_mat);

  *output = std::vector<double>(
      post_activation.data(), post_activation.data() + post_activation.size());

  *activation_gradient = std::vector<double>(
      activation_gradient_mat.data(),
      activation_gradient_mat.data() + activation_gradient_mat.size());
}

void LayerFC::BackwardPass(const std::vector<double>& input,
                           const std::vector<double>& parameters,
                           const std::vector<double>& activation_gradient,
                           const std::vector<double>& dloss_doutput,
                           std::vector<double>* dloss_dinput,
                           std::vector<double>* dloss_dparams) const {
  // Compute derivative of output with respect to input.
  // y = f(Wx + b);
  // y = f(z), where z = Wx + b.
  // dydx = dfdz*dzdx
  // dydx = f'(z) * W.

  // Layer gradient should have num_outputs_ rows and num_inputs_ cols.
  // W has                      num_outputs_ rows and num_inputs_ cols.
  // activation gradient has    num_outputs_ rows and num_outputs_ cols.

  const std::size_t num_weights = num_inputs_ * num_outputs_;
  const std::size_t num_biases = num_outputs_;

  const Eigen::Map<const Eigen::MatrixXd> activation_gradient_mat(
      activation_gradient.data(), num_outputs_, num_outputs_);

  const Eigen::Map<const Eigen::MatrixXd> W(parameters.data(), num_outputs_,
                                            num_inputs_);
  const Eigen::Map<const Eigen::VectorXd> b(parameters.data() + num_weights,
                                            num_biases);

  assert(dloss_doutput.size() == num_outputs_);
  const Eigen::Map<const Eigen::MatrixXd> dloss_doutput_mat(
      dloss_doutput.data(), 1, num_outputs_);

  const Eigen::MatrixXd doutput_dinput_mat = activation_gradient_mat * W;

  const Eigen::MatrixXd dloss_dinput_mat =
      dloss_doutput_mat * doutput_dinput_mat;

  *dloss_dinput =
      std::vector<double>(dloss_dinput_mat.data(),
                          dloss_dinput_mat.data() + dloss_dinput_mat.size());
}

int LayerConv::GetNumParameters() const {
  int num_kernel_parameters = GetNumKernelParameters();
  int num_bias_parameters = GetNumBiasParameters();
  return num_kernel_parameters + num_bias_parameters;
}

std::vector<double> LayerConv::GetRandomParameters() const {
  const double weight_coefficient = GetWeightCoefficient(activation_function_);

  // Generate weights.
  const int num_weights = GetNumKernelParameters();
  const std::vector<double> weights =
      GetRandomVector(num_weights, -1 * weight_coefficient, weight_coefficient);

  // Generate biases.
  const int num_biases = GetNumBiasParameters();
  const std::vector<double> biases =
      GetRandomVector(num_biases, -1 * weight_coefficient, weight_coefficient);

  // Output.
  std::vector<double> parameters;
  parameters.insert(parameters.end(), weights.begin(), weights.end());
  parameters.insert(parameters.end(), biases.begin(), biases.end());
  return parameters;
}

void LayerConv::ForwardPass(const std::vector<double>& input,
                            const std::vector<double>& parameters,
                            std::vector<double>* output,
                            std::vector<double>* activation_gradient) const {
  int num_kernel_parameters = GetNumKernelParameters();
  int num_bias_parameters = GetNumBiasParameters();

  assert(parameters.size() == GetNumParameters());
  assert(parameters.size() == num_kernel_parameters + num_bias_parameters);

  // TODO: Avoid copies.
  const std::vector<double> kernel_parameters(
      parameters.data(), parameters.data() + num_kernel_parameters);
  const std::vector<double> bias_parameters(
      parameters.data() + num_kernel_parameters,
      parameters.data() + num_kernel_parameters + num_bias_parameters);

  // Reshape input into InputOutputVolume.
  const InputOutputVolume input_volume(input, input_channels_, input_rows_,
                                       input_cols_);

  // Reshape parameters into ConvKernels.
  const ConvKernels conv_kernels(kernel_parameters, num_kernels_,
                                 input_channels_, kernel_rows_, kernel_cols_);

  // Perform convolution.
  std::vector<Eigen::MatrixXd> output_volume;
  std::vector<Eigen::MatrixXd> input_channels_unrolled_return;
  ConvMatrixMultiplication(input_volume.GetVolume(), conv_kernels.GetKernels(),
                           bias_parameters, padding_, stride_, &output_volume,
                           &input_channels_unrolled_return);

  // Reshape output.
  // TODO: Avoid copies.
  std::vector<double> output_values;
  for (const Eigen::MatrixXd& output_channel : output_volume) {
    const std::vector<double> output_channel_vector(
        output_channel.data(), output_channel.data() + output_channel.size());
    output_values.insert(output_values.end(), output_channel_vector.begin(),
                         output_channel_vector.end());
  }
  const Eigen::VectorXd pre_activation =
      Eigen::Map<Eigen::VectorXd>(output_values.data(), output_values.size());

  // Compute post-activation result and gradient.
  // TODO: Consider implementing Activation function to operate directly on
  // std::vector<double> input.
  Eigen::MatrixXd activation_gradient_mat;
  const Eigen::VectorXd post_activation = Activation(
      pre_activation, activation_function_, &activation_gradient_mat);

  *output = std::vector<double>(
      post_activation.data(), post_activation.data() + post_activation.size());

  *activation_gradient = std::vector<double>(
      activation_gradient_mat.data(),
      activation_gradient_mat.data() + activation_gradient_mat.size());
}

void LayerConv::BackwardPass(const std::vector<double>& input,
                             const std::vector<double>& parameters,
                             const std::vector<double>& activation_gradient,
                             const std::vector<double>& dloss_doutput,
                             std::vector<double>* dloss_dinput,
                             std::vector<double>* dloss_dparams) const {
  assert(parameters.size() == GetNumParameters());

  const std::size_t num_outputs =
      GetOutputRows() * GetOutputCols() * num_kernels_;
  assert(activation_gradient.size() == num_outputs * num_outputs);
  assert(num_outputs == dloss_doutput.size());

  std::size_t num_kernel_parameters = GetNumKernelParameters();
  std::size_t num_bias_parameters = GetNumBiasParameters();

  // TODO: Avoid copies.
  const std::vector<double> kernel_parameters(
      parameters.data(), parameters.data() + num_kernel_parameters);
  const std::vector<double> bias_parameters(
      parameters.data() + num_kernel_parameters,
      parameters.data() + num_kernel_parameters + num_bias_parameters);

  // Reshape input into InputOutputVolume.
  const InputOutputVolume input_volume(input, input_channels_, input_rows_,
                                       input_cols_);

  // Reshape parameters into ConvKernels.
  const ConvKernels conv_kernels(kernel_parameters, num_kernels_,
                                 input_channels_, kernel_rows_, kernel_cols_);

  const Eigen::VectorXd dloss_doutput_pre_act =
      Eigen::Map<const Eigen::MatrixXd>(activation_gradient.data(),
                                        dloss_doutput.size(),
                                        dloss_doutput.size()) *
      Eigen::Map<const Eigen::VectorXd>(dloss_doutput.data(),
                                        dloss_doutput.size());
  std::vector<double> dloss_doutput_pre_act_vec(
      dloss_doutput_pre_act.data(),
      dloss_doutput_pre_act.data() + dloss_doutput_pre_act.size());

  // Reshape dloss_doutput into iov format.
  const InputOutputVolume dloss_doutput_iov(dloss_doutput_pre_act_vec,
                                            num_kernels_, GetOutputRows(),
                                            GetOutputCols());

  // TODO: Still need to incorporate activation gradient!
  const InputOutputVolume activation_gradient_iov(
      activation_gradient, num_kernels_, GetOutputRows(), GetOutputCols());

  const std::vector<Eigen::MatrixXd> dloss_doutput_volume =
      dloss_doutput_iov.GetVolume();

  const std::vector<Eigen::MatrixXd> dloss_dinput_volume =
      ConvGradient(conv_kernels, dloss_doutput_volume);

  const InputOutputVolume dloss_dinput_iov(dloss_dinput_volume);
  *dloss_dinput = dloss_dinput_iov.GetValues();
}

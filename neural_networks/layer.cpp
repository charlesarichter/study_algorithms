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

int LayerConv::GetNumParameters() const {
  return kernel_rows_ * kernel_cols_ * num_kernels_ + num_kernels_;
}

std::vector<double> LayerConv::GetRandomParameters() const {
  const double weight_coefficient = GetWeightCoefficient(activation_function_);

  // Generate weights.
  const int num_weights = kernel_rows_ * kernel_cols_ * num_kernels_;
  const std::vector<double> weights =
      GetRandomVector(num_weights, -1 * weight_coefficient, weight_coefficient);

  // Generate biases.
  const std::vector<double> biases = GetRandomVector(
      num_kernels_, -1 * weight_coefficient, weight_coefficient);

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
  assert(parameters.size() == GetNumParameters());

  // TODO: Avoid copies.
  std::size_t num_kernel_parameters =
      num_kernels_ * input_channels_ * input_rows_ * input_cols_;
  std::size_t num_bias_parameters = num_kernels_;

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

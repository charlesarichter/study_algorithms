#include "layer.hpp"

#include <random>

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
  assert(parameters.size() == num_weights + num_biases);

  const Eigen::Map<const Eigen::VectorXd> input_vec(input.data(), input.size());

  // Reshape parameters into weight matrix and bias vector.
  const Eigen::Map<const Eigen::MatrixXd> W(parameters.data(), num_outputs_,
                                            num_inputs_);
  const Eigen::Map<const Eigen::VectorXd> b(parameters.data() + num_weights,
                                            num_biases);

  const Eigen::VectorXd pre_activation = W * input_vec + b;

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
                            std::vector<double>* activation_gradient) const {}

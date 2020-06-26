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
                          ActivationGradient* activation_gradient) const {
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
  const Eigen::VectorXd pre_activation_vec = W * input_vec + b;
  const std::vector<double> pre_activation(
      pre_activation_vec.data(),
      pre_activation_vec.data() + pre_activation_vec.size());

  // Compute activation and gradient.
  Activation(pre_activation, activation_function_, output, activation_gradient);
}

void LayerFC::BackwardPass(const std::vector<double>& input,
                           const std::vector<double>& parameters,
                           const ActivationGradient& activation_gradient,
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

  const Eigen::Map<const Eigen::VectorXd> input_mat(input.data(), input.size());

  const Eigen::Map<const Eigen::MatrixXd> activation_gradient_mat(
      activation_gradient.gradient.data(), num_outputs_, num_outputs_);

  const Eigen::Map<const Eigen::MatrixXd> W(parameters.data(), num_outputs_,
                                            num_inputs_);
  const Eigen::Map<const Eigen::VectorXd> b(parameters.data() + num_weights,
                                            num_biases);

  assert(dloss_doutput.size() == num_outputs_);

  // TODO: Change this to Eigen::Map<const Eigen::VectorXd> for efficency.
  const Eigen::Map<const Eigen::MatrixXd> dloss_doutput_mat(
      dloss_doutput.data(), 1, num_outputs_);

  // TODO: Toggle computation based on whether activation_gradient_mat is
  // diagonal. See LayerConv::BackwardPass for example.
  const Eigen::MatrixXd doutput_dinput_mat = activation_gradient_mat * W;
  const Eigen::MatrixXd dloss_dweights_mat = activation_gradient_mat *
                                             dloss_doutput_mat.transpose() *
                                             input_mat.transpose();
  const Eigen::MatrixXd dloss_dbiases_mat =
      activation_gradient_mat * dloss_doutput_mat.transpose();

  const Eigen::MatrixXd dloss_dinput_mat =
      dloss_doutput_mat * doutput_dinput_mat;

  *dloss_dinput =
      std::vector<double>(dloss_dinput_mat.data(),
                          dloss_dinput_mat.data() + dloss_dinput_mat.size());

  const std::vector<double> dloss_dweights_vec(
      dloss_dweights_mat.data(),
      dloss_dweights_mat.data() + dloss_dweights_mat.size());
  const std::vector<double> dloss_dbiases_vec(
      dloss_dbiases_mat.data(),
      dloss_dbiases_mat.data() + dloss_dbiases_mat.size());

  assert(dloss_dweights_vec.size() + dloss_dbiases_vec.size() ==
         parameters.size());

  dloss_dparams->clear();
  dloss_dparams->insert(dloss_dparams->end(), dloss_dweights_vec.begin(),
                        dloss_dweights_vec.end());
  dloss_dparams->insert(dloss_dparams->end(), dloss_dbiases_vec.begin(),
                        dloss_dbiases_vec.end());
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
                            ActivationGradient* activation_gradient) const {
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

  // Compute activation and gradient.
  Activation(output_values, activation_function_, output, activation_gradient);
}

void LayerConv::BackwardPass(const std::vector<double>& input,
                             const std::vector<double>& parameters,
                             const ActivationGradient& activation_gradient,
                             const std::vector<double>& dloss_doutput,
                             std::vector<double>* dloss_dinput,
                             std::vector<double>* dloss_dparams) const {
  assert(parameters.size() == GetNumParameters());

  const std::size_t num_outputs =
      GetOutputRows() * GetOutputCols() * num_kernels_;
  assert(activation_gradient.gradient.size() == num_outputs * num_outputs);
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

  // Toggle computation based on whether activation_gradient is diagonal or not.
  Eigen::VectorXd dloss_doutput_pre_act;
  if (activation_gradient.diagonal) {
    dloss_doutput_pre_act =
        Eigen::Map<const Eigen::MatrixXd>(activation_gradient.gradient.data(),
                                          dloss_doutput.size(),
                                          dloss_doutput.size())
            .diagonal()
            .cwiseProduct(Eigen::Map<const Eigen::VectorXd>(
                dloss_doutput.data(), dloss_doutput.size()));
  } else {
    dloss_doutput_pre_act = Eigen::Map<const Eigen::MatrixXd>(
                                activation_gradient.gradient.data(),
                                dloss_doutput.size(), dloss_doutput.size()) *
                            Eigen::Map<const Eigen::VectorXd>(
                                dloss_doutput.data(), dloss_doutput.size());
  }

  std::vector<double> dloss_doutput_pre_act_vec(
      dloss_doutput_pre_act.data(),
      dloss_doutput_pre_act.data() + dloss_doutput_pre_act.size());

  const int num_biases = GetNumBiasParameters();
  assert(dloss_doutput_pre_act_vec.size() % num_biases == 0);
  const int num_outputs_per_bias =
      dloss_doutput_pre_act_vec.size() / num_biases;

  const Eigen::Map<const Eigen::MatrixXd> dloss_doutput_pre_act_mat(
      dloss_doutput_pre_act_vec.data(), num_outputs_per_bias, num_biases);

  const Eigen::VectorXd bias_gradient_vec =
      dloss_doutput_pre_act_mat.colwise().sum();
  assert(bias_gradient_vec.size() == num_biases);

  // Reshape dloss_doutput into iov format.
  const InputOutputVolume dloss_doutput_iov(dloss_doutput_pre_act_vec,
                                            num_kernels_, GetOutputRows(),
                                            GetOutputCols());

  const std::vector<Eigen::MatrixXd> dloss_doutput_volume =
      dloss_doutput_iov.GetVolume();

  const std::vector<Eigen::MatrixXd> dloss_dinput_volume =
      ConvGradient(conv_kernels, dloss_doutput_volume);

  const std::vector<Eigen::MatrixXd> dloss_dweights_volume =
      ConvWeightGradient(input_volume.GetVolume(), dloss_doutput_volume);

  const InputOutputVolume dloss_dinput_iov(dloss_dinput_volume);
  *dloss_dinput = dloss_dinput_iov.GetValues();

  const InputOutputVolume dloss_dweights_iov(dloss_dweights_volume);
  *dloss_dparams = dloss_dweights_iov.GetValues();

  const std::vector<double> dloss_dbiases(
      bias_gradient_vec.data(), bias_gradient_vec.data() + num_biases);

  dloss_dparams->insert(dloss_dparams->end(), dloss_dbiases.begin(),
                        dloss_dbiases.end());
}

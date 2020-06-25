#include "network.hpp"

#include <chrono>
#include <iostream>

std::vector<double> Network::GetRandomParameters() const {
  std::vector<double> parameters;
  for (const LayerPtr& layer : layers_) {
    std::vector<double> layer_parameters = layer->GetRandomParameters();
    parameters.insert(parameters.end(), layer_parameters.begin(),
                      layer_parameters.end());
  }
  return parameters;
}

std::vector<double> Network::Evaluate(
    const std::vector<double>& input, const std::vector<double>& label,
    const std::vector<double>& parameters) const {
  // Iterator indicating the beginning of the current layer's parameters.
  auto param_begin = parameters.begin();

  // Initialize input.
  std::vector<double> layer_input = input;

  // Foward pass.
  for (const LayerPtr& layer : layers_) {
    // Get parameters for this layer. TODO: Reduce/avoid copies.
    const int num_params = layer->GetNumParameters();
    const std::vector<double> layer_param(param_begin,
                                          param_begin + num_params);

    // Evaluate layer.
    std::vector<double> layer_output;
    std::vector<double> layer_activation_gradient;
    layer->ForwardPass(layer_input, layer_param, &layer_output,
                       &layer_activation_gradient);

    // Copy output to next layer's input.
    layer_input = layer_output;

    // Advance the param iterator.
    param_begin += num_params;
  }

  return layer_input;
}

double Network::Evaluate(const std::vector<double>& input,
                         const std::vector<double>& label,
                         const std::vector<double>& parameters,
                         std::vector<double>* input_gradient,
                         std::vector<double>* param_gradient,
                         NetworkTiming* timing) const {
  auto forward_pass_start = std::chrono::steady_clock::now();

  // Iterator indicating the beginning of the current layer's parameters.
  auto param_begin = parameters.begin();

  // Initialize input.
  std::vector<double> layer_input = input;

  // Store the activation gradients from each layer during the forward pass.
  std::vector<std::vector<double>> layer_activation_gradients;

  // Store the inputs.
  std::vector<std::vector<double>> layer_inputs;

  // Store the parameters.
  std::vector<std::vector<double>> layer_params;

  // Foward pass.
  for (const LayerPtr& layer : layers_) {
    // Get parameters for this layer. TODO: Reduce/avoid copies.
    const int num_params = layer->GetNumParameters();
    const std::vector<double> layer_param(param_begin,
                                          param_begin + num_params);

    // Evaluate layer.
    std::vector<double> layer_output;
    std::vector<double> layer_activation_gradient;
    layer->ForwardPass(layer_input, layer_param, &layer_output,
                       &layer_activation_gradient);

    // Store activation gradient. TODO: Reduce/avoid copies.
    layer_inputs.emplace_back(layer_input);
    layer_params.emplace_back(layer_param);
    layer_activation_gradients.emplace_back(layer_activation_gradient);

    // Copy output to next layer's input.
    layer_input = layer_output;

    // Advance the param iterator.
    param_begin += num_params;
  }

  // Evaluate loss.
  const Eigen::VectorXd label_vector =
      Eigen::Map<const Eigen::VectorXd>(label.data(), label.size());
  const Eigen::VectorXd network_output =
      Eigen::Map<Eigen::VectorXd>(layer_input.data(), layer_input.size());

  Eigen::VectorXd loss_gradient;
  const Eigen::VectorXd loss =
      Loss(network_output, label_vector, LossFunction::CROSS_ENTROPY,
           &loss_gradient);
  assert(loss.size() == 1);

  auto forward_pass_end = std::chrono::steady_clock::now();
  auto backward_pass_start = std::chrono::steady_clock::now();

  // Backward pass.

  // loss_gradient is the derivative of the loss output with respect to its
  // inputs (i.e., the network output).

  std::vector<double> dloss_dnetwork(
      loss_gradient.data(), loss_gradient.data() + loss_gradient.size());
  // std::cerr << "dloss_dnetwork size: " << dloss_dnetwork.size() << std::endl;

  std::vector<std::vector<double>> layer_param_gradients;

  for (int i = (layers_.size() - 1); i >= 0; --i) {
    const LayerPtr& layer = layers_.at(i);

    // TODO: Avoid reusing variable names.
    const std::vector<double>& layer_input = layer_inputs.at(i);
    const std::vector<double>& layer_param = layer_params.at(i);
    const std::vector<double>& layer_act_grad =
        layer_activation_gradients.at(i);

    std::vector<double> dloss_dnetwork_updated;
    std::vector<double> dloss_dparams;

    layer->BackwardPass(layer_input, layer_param, layer_act_grad,
                        dloss_dnetwork, &dloss_dnetwork_updated,
                        &dloss_dparams);
    dloss_dnetwork = dloss_dnetwork_updated;
    // std::cerr << "dloss_dnetwork size: " << dloss_dnetwork.size() <<
    // std::endl;

    layer_param_gradients.emplace_back(dloss_dparams);
  }

  // Return layer param gradients in correct order.
  for (int i = (layer_param_gradients.size() - 1); i >= 0; --i) {
    const std::vector<double>& layer_param_gradient =
        layer_param_gradients.at(i);
    param_gradient->insert(param_gradient->end(), layer_param_gradient.begin(),
                           layer_param_gradient.end());
  }

  *input_gradient = dloss_dnetwork;

  // std::cerr << "Num param gradients: " << param_gradient->size() <<
  // std::endl; std::cerr << "Num params: " << parameters.size() << std::endl;
  assert(param_gradient->size() == parameters.size());

  auto backward_pass_end = std::chrono::steady_clock::now();
  std::chrono::duration<double> forward_pass_elapsed_seconds =
      forward_pass_end - forward_pass_start;
  std::chrono::duration<double> backward_pass_elapsed_seconds =
      backward_pass_end - backward_pass_start;
  timing->forward_pass = forward_pass_elapsed_seconds.count();
  timing->backward_pass = backward_pass_elapsed_seconds.count();

  return loss(0);  // Make loss a scalar.
}

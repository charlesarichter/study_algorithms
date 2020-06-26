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

std::vector<double> Network::Evaluate(const std::vector<double>& input,
                                      const std::vector<double>& label,
                                      const std::vector<double>& parameters) {
  // TODO: Pre-allocate these containers and assert that their size is correct.
  if (layer_io_.size() != (layers_.size() + 1)) {
    std::cerr << "Re-sizing layer IO container" << std::endl;
    layer_io_.resize(layers_.size() + 1);
  }
  if (layer_activation_gradients_.size() != (layers_.size() + 1)) {
    std::cerr << "Re-sizing layer activation gradients container" << std::endl;
    layer_activation_gradients_.resize(layers_.size() + 1);
  }

  // Iterator indicating the beginning of the current layer's parameters.
  auto param_begin = parameters.begin();

  // Initialize input.
  layer_io_.front() = input;

  // Foward pass.
  for (std::size_t i = 0; i < layers_.size(); ++i) {
    const LayerPtr& layer = layers_.at(i);

    // Get parameters for this layer. TODO: Reduce/avoid copies.
    const int num_params = layer->GetNumParameters();
    const std::vector<double> layer_param(param_begin,
                                          param_begin + num_params);

    // Evaluate layer.
    layer->ForwardPass(layer_io_.at(i), layer_param, &layer_io_.at(i + 1),
                       &layer_activation_gradients_.at(i));

    // Advance the param iterator.
    param_begin += num_params;
  }

  // TODO: Can also use layers_.back() if you confirm size is correct.
  return layer_io_.at(layers_.size());
}

double Network::Evaluate(const std::vector<double>& input,
                         const std::vector<double>& label,
                         const std::vector<double>& parameters,
                         std::vector<double>* input_gradient,
                         std::vector<double>* param_gradient,
                         NetworkTiming* timing) {
  auto forward_pass_start = std::chrono::steady_clock::now();

  // Iterator indicating the beginning of the current layer's parameters.
  auto param_begin = parameters.begin();

  // TODO: Pre-allocate these containers and assert that their size is correct.
  if (layer_io_.size() != (layers_.size() + 1)) {
    std::cerr << "Re-sizing layer IO container" << std::endl;
    layer_io_.resize(layers_.size() + 1);
  }
  if (layer_activation_gradients_.size() != (layers_.size() + 1)) {
    std::cerr << "Re-sizing layer activation gradients container" << std::endl;
    layer_activation_gradients_.resize(layers_.size() + 1);
  }

  // Initialize input.
  layer_io_.front() = input;

  // Store the parameters.
  std::vector<std::vector<double>> layer_params;

  // Foward pass.
  for (std::size_t i = 0; i < layers_.size(); ++i) {
    const LayerPtr& layer = layers_.at(i);

    // Get parameters for this layer. TODO: Reduce/avoid copies.
    const int num_params = layer->GetNumParameters();
    const std::vector<double> layer_param(param_begin,
                                          param_begin + num_params);

    // Evaluate layer.
    layer->ForwardPass(layer_io_.at(i), layer_param, &layer_io_.at(i + 1),
                       &layer_activation_gradients_.at(i));

    // TODO: Reduce/avoid copies.
    layer_params.emplace_back(layer_param);

    // Advance the param iterator.
    param_begin += num_params;
  }

  // Evaluate loss.
  const Eigen::VectorXd label_vector =
      Eigen::Map<const Eigen::VectorXd>(label.data(), label.size());

  // TODO: Can also use layers_.back() if you confirm size is correct.
  const Eigen::VectorXd network_output = Eigen::Map<Eigen::VectorXd>(
      layer_io_.at(layers_.size()).data(), layer_io_.at(layers_.size()).size());

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
    const std::vector<double>& layer_input = layer_io_.at(i);
    const std::vector<double>& layer_param = layer_params.at(i);
    const ActivationGradient& layer_act_grad =
        layer_activation_gradients_.at(i);

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

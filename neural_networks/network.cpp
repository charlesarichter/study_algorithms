#include "network.hpp"

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

double Network::Evaluate(const std::vector<double>& input,
                         const std::vector<double>& label,
                         const std::vector<double>& parameters,
                         std::vector<double>* gradient) {
  // Iterator indicating the beginning of the current layer's parameters.
  auto param_begin = parameters.begin();

  // Initialize input.
  std::vector<double> layer_input = input;

  // Store the activation gradients from each layer during the forward pass.
  std::vector<std::vector<double>> layer_activation_gradients;

  std::cerr << "Evaluating..." << std::endl;

  for (const LayerPtr& layer : layers_) {
    // Get parameters for this layer. TODO: Reduce/avoid copies.
    const int num_params = layer->GetNumParameters();
    const std::vector<double> layer_parameters(param_begin,
                                               param_begin + num_params);

    std::cerr << "Layer input size: " << layer_input.size() << std::endl;

    // Evaluate layer.
    std::vector<double> layer_output;
    std::vector<double> layer_activation_gradient;
    layer->ForwardPass(layer_input, layer_parameters, &layer_output,
                       &layer_activation_gradient);

    std::cerr << "Layer output size: " << layer_output.size() << std::endl;

    // Store activation gradient. TODO: Reduce/avoid copies.
    layer_activation_gradients.emplace_back(layer_activation_gradient);

    // Copy output to next layer's input.
    layer_input = layer_output;

    // Advance the param iterator.
    param_begin += num_params;
  }

  return 0;
}

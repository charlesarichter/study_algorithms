#pragma once

#include "layer.hpp"

struct NetworkTiming {
  double forward_pass = 0;
  double backward_pass = 0;
};

class Network {
 public:
  Network(const std::vector<LayerPtr>& layers) : layers_(layers){};

  /**
   * Returns a random initialization of network parameters with initialization
   * of each parameter appropriate for each layer.
   */
  std::vector<double> GetRandomParameters() const;

  /**
   * Evaluates network using provided input and parameters and returns loss and
   * loss gradient.
   */
  double Evaluate(const std::vector<double>& input,
                  const std::vector<double>& label,
                  const std::vector<double>& parameters,
                  std::vector<double>* input_gradient,
                  std::vector<double>* param_gradient, NetworkTiming* timing);

  /**
   * Evaluates network using provided input and parameters but does not compute
   * or return gradients.
   */
  std::vector<double> Evaluate(const std::vector<double>& input,
                               const std::vector<double>& label,
                               const std::vector<double>& parameters);

 private:
  std::vector<LayerPtr> layers_;

  // TODO: Compute and pre-allocate the sizes of these containers and their
  // elements, then assert that that size is correct everywhere they are used.
  std::vector<std::vector<double>> layer_io;
  std::vector<std::vector<double>> layer_activation_gradients;
};

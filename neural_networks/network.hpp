#pragma once

#include "layer.hpp"

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
                  std::vector<double>* param_gradient) const;

  /**
   * Evaluates network using provided input and parameters but does not compute
   * or return gradients.
   */
  std::vector<double> Evaluate(const std::vector<double>& input,
                               const std::vector<double>& label,
                               const std::vector<double>& parameters) const;

 private:
  std::vector<LayerPtr> layers_;
};

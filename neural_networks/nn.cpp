#include "nn.hpp"

#include <iostream>

NeuralNetworkParameters GetRandomNeuralNetwork(
    int input_dimension, int output_dimension, int num_hidden_layers,
    int nodes_per_hidden_layer, const ActivationFunction hidden_activation,
    const ActivationFunction output_activation) {
  NeuralNetworkParameters nn;
  // Input layer
  nn.weights.emplace_back(
      Eigen::MatrixXd::Random(nodes_per_hidden_layer, input_dimension));
  nn.biases.emplace_back(Eigen::VectorXd::Random(nodes_per_hidden_layer));
  nn.activation_functions.emplace_back(hidden_activation);

  // Randomly initialize weights and biases for each layer
  for (int i = 0; i < num_hidden_layers - 1; ++i) {
    nn.weights.emplace_back(Eigen::MatrixXd::Random(nodes_per_hidden_layer,
                                                    nodes_per_hidden_layer));
    nn.biases.emplace_back(Eigen::VectorXd::Random(nodes_per_hidden_layer));
    nn.activation_functions.emplace_back(hidden_activation);
  }

  // Output layer
  nn.weights.emplace_back(
      Eigen::MatrixXd::Random(output_dimension, nodes_per_hidden_layer));
  nn.biases.emplace_back(Eigen::VectorXd::Random(output_dimension));
  nn.activation_functions.emplace_back(output_activation);
  return nn;
}

// Example derivation of gradients:
//
// Consider a single hidden layer
// y = f1(A1*f0(A0*x0 + b0) + b1)
//
// A0 and A1 are weight matrices
// b0 and b1 are bias vectors
//
// want derivative of y with respect to A0, A1, b0, b1, *evaluated at x0*
//
// dy/dA1 = f1'(A1*f0(A0*x0 + b0) + b1) * f0(A0*x0 + b0)
// dy/db1 = f1'(A1*f0(A0*x0 + b0) + b1)
// dy/dA0 = f1'(A1*f0(A0*x0 + b0) + b1) * A1 * f0'(A0*x0 + b0) * x0
// dy/db0 = f1'(A1*f0(A0*x0 + b0) + b1) * A1 * f0'(A0*x0 + b0)
//
// Written differently, where x1 = f0(A0*x0 + b0),
// dy/dA1 = f1'(A1*x1 + b1) * x1
// dy/db1 = f1'(A1*x1 + b1)
// dy/dA0 = f1'(A1*x1 + b1) * A1 * f0'(A0*x0 + b0) * x0
// dy/db0 = f1'(A1*x1 + b1) * A1 * f0'(A0*x0 + b0)

// Example derivation of gradients:
//
// Consider a single hidden layer with linear activations
// y = A1*(A0*x0 + b0) + b1
// dy/dA1 = A0*x0 + b0
// dy/db1 = I

void EvaluateNetwork(const Eigen::VectorXd& input,
                     const NeuralNetworkParameters& params,
                     Eigen::VectorXd* output,
                     std::vector<Eigen::MatrixXd>* weight_gradients,
                     std::vector<Eigen::VectorXd>* bias_gradients) {
  // Forward pass
  Eigen::VectorXd current_value = input;
  std::vector<Eigen::VectorXd> activation_gradients;
  std::vector<Eigen::VectorXd> post_activation_results;
  post_activation_results.emplace_back(input);
  for (int i = 0; i < params.weights.size(); ++i) {
    // Compute pre-activation input.
    const Eigen::VectorXd pre_activation =
        params.weights.at(i) * current_value + params.biases.at(i);

    // Compute activation output.
    Eigen::VectorXd activation_gradient;
    current_value =
        Activation(pre_activation, params.activation_functions.at(i),
                   &activation_gradient);
    post_activation_results.emplace_back(current_value);
    activation_gradients.emplace_back(activation_gradient);
  }
  *output = current_value;

  // Backward pass

  // TODO: Dimensionality of the output? Determine the dimensionality from the
  // network structure. Don't hardcode it.
  Eigen::MatrixXd a = Eigen::MatrixXd::Ones(1, 1);

  // Allocate output.
  weight_gradients->resize(params.weights.size());
  bias_gradients->resize(params.weights.size());

  for (int i = (params.weights.size() - 1); i >= 0; --i) {
    const Eigen::MatrixXd dydb =
        a.cwiseProduct(activation_gradients.at(i).transpose()).transpose();
    const Eigen::MatrixXd dydw =
        dydb * post_activation_results.at(i).transpose();
    a = a.cwiseProduct(activation_gradients.at(i).transpose()) *
        params.weights.at(i);

    weight_gradients->at(i) = dydw;
    bias_gradients->at(i) = dydb;
  }
}

Eigen::VectorXd Activation(const Eigen::VectorXd& input,
                           const ActivationFunction activation_function,
                           Eigen::VectorXd* activation_gradient) {
  Eigen::VectorXd output(input.size());
  switch (activation_function) {
    case ActivationFunction::LINEAR: {
      output = input;

      // Slope of one.
      *activation_gradient = Eigen::VectorXd::Ones(input.size());
      break;
    }
    case ActivationFunction::SIGMOID: {
      *activation_gradient = Eigen::VectorXd::Zero(input.size());

      // TODO: More efficient/vectorized computation.
      for (size_t i = 0; i < input.size(); ++i) {
        const double f = 1 / (1 + exp(-1 * input(i)));
        output(i) = f;
        (*activation_gradient)(i) = f * (1 - f);
      }

      break;
    }
    case ActivationFunction::RELU: {
      break;
    }
    default: {
      throw std::runtime_error("Invalid activation type.");
      break;
    }
  }
  return output;
}

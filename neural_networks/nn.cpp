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

// NeuralNetwork::~NeuralNetwork() {}

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
  // std::cerr << "Input: " << current_value << std::endl;

  // Forward pass
  Eigen::VectorXd current_value = input;
  std::vector<Eigen::VectorXd> pre_activation_results;
  std::vector<Eigen::VectorXd> post_activation_results;
  std::vector<Eigen::VectorXd> activation_gradients;
  for (int i = 0; i < params.weights.size(); ++i) {
    // Compute pre-activation input.
    const Eigen::VectorXd pre_activation =
        params.weights.at(i) * current_value + params.biases.at(i);
    pre_activation_results.emplace_back(pre_activation);

    // Compute activation output.
    Eigen::VectorXd activation_gradient;
    current_value =
        Activation(pre_activation, params.activation_functions.at(i),
                   &activation_gradient);
    post_activation_results.emplace_back(current_value);
    activation_gradients.emplace_back(activation_gradient);

    // std::cerr << "i: " << i << std::endl;
    // std::cerr << "Weight: " << weights_.at(i) << std::endl;
    // std::cerr << "Bias: " << biases_.at(i) << std::endl;
    // std::cerr << "Pre-activation: " << pre_activation << std::endl;
    // std::cerr << "Post-activation: " << current_value << std::endl;
  }
  *output = current_value;

  // Backward pass (backprop)

  // dy/dA1 = f1'(layer_1_pre_act) * layer_0_post_act
  // dy/dA0 = f1'(layer_1_pre_act) * A1 * f0'(layer_0_pre_act) * input

  // dy/dA2 = f2'(layer_2_pre_act) * layer_1_post_act
  // dy/dA1 = f2'(layer_2_pre_act) * A2 * f1'(layer_1_pre_act) * layer_0_post_act
  // dy/dA0 = f2'(layer_2_pre_act) * A2 * f1'(layer_1_pre_act) * A1 * f0'(layer_0_pre_act) * input

  // // Manually compute it here to quickly verify
  // Eigen::MatrixXd dydA1 = post_activation_results.at(0).transpose();
  // // Eigen::MatrixXd dydb1 = Eigen::MatrixXd::Identity(
  // Eigen::MatrixXd dydA0 = params.weights.at(1).transpose() * input;
  // std::cerr << "Gradient dydA1..." << dydA1 << std::endl;
  // std::cerr << "Gradient dydA0..." << dydA0.transpose() << std::endl;

  // Attempt at implementing backprop.
  //
  // Copied from comments above:
  // Written differently, where x1 = f0(A0*x0 + b0),
  // dy/dA1 = f1'(A1*x1 + b1) * x1
  // dy/db1 = f1'(A1*x1 + b1)
  // dy/dA0 = f1'(A1*x1 + b1) * A1 * f0'(A0*x0 + b0) * x0
  // dy/db0 = f1'(A1*x1 + b1) * A1 * f0'(A0*x0 + b0)

  // Eigen::MatrixXd dydA1_test =
  //     activation_gradients.at(1) * post_activation_results.at(0).transpose();
  // std::cerr << "Gradient dydA1_test..." << dydA1_test << std::endl;

  // Eigen::MatrixXd dydA0_test = activation_gradients.at(1) *
  //                              params.weights.at(1) *
  //                              activation_gradients.at(0) * input;

  // 2 x 3 or 3 x 2: activation_gradients.at(0) * input.transpose();
  // 3 x 1 or 1 x 3: params.weights.at(1);
  //
  // Eigen::MatrixXd dydA0_test =
  //     activation_gradients.at(1) *
  //     (params.weights.at(1) * (activation_gradients.at(0) *
  //     input.transpose()));
  // std::cerr << "Gradient dydA0_test..." << dydA0_test << std::endl;

  // std::cerr << "1: " << activation_gradients.at(1) << std::endl;
  // std::cerr << "2: " << params.weights.at(1) << std::endl;
  // std::cerr << "3: " << activation_gradients.at(0) << std::endl;
  // std::cerr << "4: " << input << std::endl;
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

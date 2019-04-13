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

void EvaluateNetworkLoss(const Eigen::VectorXd& input,
                         const NeuralNetworkParameters& params,
                         const Eigen::VectorXd& label,
                         const LossFunction& loss_function,
                         Eigen::VectorXd* loss,
                         std::vector<Eigen::MatrixXd>* weight_gradients,
                         std::vector<Eigen::VectorXd>* bias_gradients) {
  // Compute output and gradients of the network itself.
  Eigen::VectorXd network_output;
  std::vector<Eigen::MatrixXd> network_weight_gradients;
  std::vector<Eigen::VectorXd> network_bias_gradients;
  EvaluateNetwork(input, params, &network_output, &network_weight_gradients,
                  &network_bias_gradients);

  // Compute the loss.
  Eigen::VectorXd loss_gradient;
  *loss = Loss(network_output, label, loss_function, &loss_gradient);

  // Compute gradients of loss w.r.t. network params.
  // TODO(charlie-or): Make this N-dimensional. In order to do that, you would
  // have to have multi-output support everywhere else in the code. Instead of
  // the derivative of a single output with respect to each weight in a layer
  // (i.e., 2D matrix) you would have to have the derivative of each output with
  // respect to each weight in a layer (i.e., a stack of 2D matrices, or a 3D
  // tensor). For now, we just assume that we have a scalar output and scalar
  // loss function and assert that its dimension is 1.
  for (size_t i = 0; i < network_weight_gradients.size(); ++i) {
    assert(loss_gradient.size() == 1);
    const double loss_gradient_1d = loss_gradient(0);
    weight_gradients->emplace_back(network_weight_gradients.at(i) *
                                   loss_gradient_1d);
  }
  for (size_t i = 0; i < network_bias_gradients.size(); ++i) {
    assert(loss_gradient.size() == 1);
    const double loss_gradient_1d = loss_gradient(0);
    bias_gradients->emplace_back(network_bias_gradients.at(i) *
                                 loss_gradient_1d);
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

Eigen::VectorXd Loss(const Eigen::VectorXd& input, const Eigen::VectorXd& label,
                     const LossFunction loss_function,
                     Eigen::VectorXd* loss_gradient) {
  switch (loss_function) {
    case LossFunction::CROSS_ENTROPY: {
      // TODO: Make this N-dimensional.
      assert(label.size() == 1);
      assert(input.size() == 1);

      const double label_1d = label(0);
      const double p_predicted_1d = input(0);
      const double loss_1d = -1 * (label_1d * log(p_predicted_1d) +
                                   (1 - label_1d) * log(1 - p_predicted_1d));
      const Eigen::VectorXd loss = loss_1d * Eigen::VectorXd::Ones(1);

      // Compute gradients.
      const double dloss_dpredicted_1d = -label_1d * (1 / p_predicted_1d) -
                                         (label_1d - 1) / (1 - p_predicted_1d);
      *loss_gradient = dloss_dpredicted_1d * Eigen::VectorXd::Ones(1);
      return loss;
      break;
    }
    default: {
      throw std::runtime_error("Unsupported loss function");
      break;
    }
  }
}

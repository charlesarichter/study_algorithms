#pragma once

#include <eigen3/Eigen/Dense>
#include <vector>

// NOTE: See https://en.wikipedia.org/wiki/activation_function
enum class ActivationFunction { LINEAR, RELU, SIGMOID, SOFTMAX };

enum class LossFunction { SQUARED_ERROR, CROSS_ENTROPY };

struct NeuralNetworkParameters {
  std::vector<Eigen::MatrixXd> weights;
  std::vector<Eigen::VectorXd> biases;
  std::vector<ActivationFunction> activation_functions;
};

double GetWeightCoefficient(const ActivationFunction& activation_function);

std::vector<double> GetRandomVector(const int num_elements,
                                    const double min_value,
                                    const double max_value);

NeuralNetworkParameters GetRandomNeuralNetwork(
    int input_dimension, int output_dimension, int num_hidden_layers,
    int nodes_per_hidden_layer, const ActivationFunction hidden_activation,
    const ActivationFunction output_activation);

void EvaluateNetwork(
    const Eigen::VectorXd& input, const NeuralNetworkParameters& params,
    Eigen::VectorXd* output,
    std::vector<std::vector<Eigen::MatrixXd>>* weight_gradients,
    std::vector<std::vector<Eigen::VectorXd>>* bias_gradients);

void EvaluateNetworkLoss(const Eigen::VectorXd& input,
                         const NeuralNetworkParameters& params,
                         const Eigen::VectorXd& label,
                         const LossFunction& loss_function,
                         Eigen::VectorXd* loss,
                         std::vector<Eigen::MatrixXd>* weight_gradients,
                         std::vector<Eigen::VectorXd>* bias_gradients);

// EvaluateNetworkLossCombinedImplementation combines EvaluateNetwork and Loss
// in a way that avoids having to compute the gradient of *each* network output
// channel (e.g., 10 channels in a softmax classifier) w.r.t. each of the
// weights and biases before combining those resuts in the loss gradient
// calculation. Effectively, computing the loss in the same place as the network
// output enables the overall output dimension to be 1 for the purposes of
// relevant gradient computation, saving a lot of computation. Tests should
// confirm that it's equivalent to EvaluateNetworkLoss.
void EvaluateNetworkLossCombinedImplementation(
    const Eigen::VectorXd& input, const NeuralNetworkParameters& params,
    const Eigen::VectorXd& label, const LossFunction& loss_function,
    Eigen::VectorXd* loss, std::vector<Eigen::MatrixXd>* weight_gradients,
    std::vector<Eigen::VectorXd>* bias_gradients);

Eigen::VectorXd Activation(const Eigen::VectorXd& input,
                           const ActivationFunction activation_function,
                           Eigen::MatrixXd* activation_gradient);

// TODO: Create a test to enforce that this implementation produces exactly
// identical results to the Eigen-based one above.
// TODO: After confirming equivalence, make the Eigen-based version an overload
// that interally calls this implementation.
void Activation(const std::vector<double>& input,
                const ActivationFunction activation_function,
                std::vector<double>* activation,
                std::vector<double>* activation_gradient);

// TODO: Make Loss output a scalar...not sure why this ever returned a vector.
Eigen::VectorXd Loss(const Eigen::VectorXd& input, const Eigen::VectorXd& label,
                     const LossFunction loss_function,
                     Eigen::VectorXd* loss_gradient);

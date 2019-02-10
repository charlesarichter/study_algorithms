#include <iostream>

#include "nn.hpp"

void ComputeGradientLayerZeroWeightsLinear(const NeuralNetworkParameters& nn,
                                           const Eigen::VectorXd& input) {
  const size_t layer = 0;
  const Eigen::MatrixXd& w0 = nn.weights.at(layer);
  // std::cerr << "Layer Zero Input: " << input.transpose() << std::endl;
  // std::cerr << "Layer Zero Weights 0: " << std::endl << w0 << std::endl;

  // Compute network output for the original network.
  Eigen::VectorXd output;
  EvaluateNetwork(input, nn, &output);

  // Gradient matrix to be filled in.
  Eigen::MatrixXd dydw0 = Eigen::MatrixXd::Zero(w0.rows(), w0.cols());
  Eigen::MatrixXd dydw0_numerical = Eigen::MatrixXd::Zero(w0.rows(), w0.cols());
  Eigen::MatrixXd dydw0_vectorized =
      Eigen::MatrixXd::Zero(w0.rows(), w0.cols());

  const double dw = 1e-6;
  for (size_t i = 0; i < w0.rows(); ++i) {
    for (size_t j = 0; j < w0.cols(); ++j) {
      // Perturb this particular element of the weight matrix.
      NeuralNetworkParameters nn_perturbation = nn;
      nn_perturbation.weights.at(layer)(i, j) += dw;

      // Compute network output for the perturbed network.
      Eigen::VectorXd output_perturbation;
      EvaluateNetwork(input, nn_perturbation, &output_perturbation);

      // Compute scalar value of numerical gradient.
      const Eigen::VectorXd vector_grad = (output_perturbation - output) / dw;
      assert(vector_grad.size() == 1);
      const double scalar_grad = vector_grad(0);
      dydw0_numerical(i, j) = scalar_grad;
    }
  }

  // Manually compute some gradients to verify for now, utilizing the fact that
  // we're using a linear activation function.
  dydw0(0, 0) = nn.weights.at(1)(0, 0) * input(0);  // dy/da_00^0
  dydw0(0, 1) = nn.weights.at(1)(0, 0) * input(1);  // dy/da_01^0
  dydw0(1, 0) = nn.weights.at(1)(0, 1) * input(0);  // dy/da_10^0
  dydw0(1, 1) = nn.weights.at(1)(0, 1) * input(1);  // dy/da_11^0
  dydw0(2, 0) = nn.weights.at(1)(0, 2) * input(0);  // dy/da_20^0
  dydw0(2, 1) = nn.weights.at(1)(0, 2) * input(1);  // dy/da_21^0

  // Vectorized
  dydw0_vectorized = nn.weights.at(1).transpose() * input.transpose();

  std::cerr << "Numerical Gradient: " << std::endl
            << dydw0_numerical << std::endl;
  std::cerr << "Analytical Gradient: " << std::endl << dydw0 << std::endl;
  std::cerr << "Vectorized Gradient: " << std::endl
            << dydw0_vectorized << std::endl;
  std::cerr << "Difference of Gradients: " << std::endl
            << dydw0 - dydw0_numerical << std::endl;
}

void ComputeGradientLayerOneWeightsLinear(const NeuralNetworkParameters& nn,
                                          const Eigen::VectorXd& input) {
  const size_t layer = 1;
  const Eigen::MatrixXd& w1 = nn.weights.at(layer);
  // std::cerr << "Layer Zero Input: " << input.transpose() << std::endl;
  // std::cerr << "Layer Zero Weights 0: " << std::endl << w1 << std::endl;

  // Compute network output for the original network.
  Eigen::VectorXd output;
  EvaluateNetwork(input, nn, &output);

  // Gradient matrix to be filled in.
  Eigen::MatrixXd dydw1 = Eigen::MatrixXd::Zero(w1.rows(), w1.cols());
  Eigen::MatrixXd dydw1_numerical = Eigen::MatrixXd::Zero(w1.rows(), w1.cols());
  Eigen::MatrixXd dydw1_vectorized =
      Eigen::MatrixXd::Zero(w1.rows(), w1.cols());

  const double dw = 1e-6;
  for (size_t i = 0; i < w1.rows(); ++i) {
    for (size_t j = 0; j < w1.cols(); ++j) {
      // Perturb this particular element of the weight matrix.
      NeuralNetworkParameters nn_perturbation = nn;
      nn_perturbation.weights.at(layer)(i, j) += dw;

      // Compute network output for the perturbed network.
      Eigen::VectorXd output_perturbation;
      EvaluateNetwork(input, nn_perturbation, &output_perturbation);

      // Compute scalar value of numerical gradient.
      const Eigen::VectorXd vector_grad = (output_perturbation - output) / dw;
      assert(vector_grad.size() == 1);
      const double scalar_grad = vector_grad(0);
      dydw1_numerical(i, j) = scalar_grad;
    }
  }

  // Manually compute some gradients to verify for now, utilizing the fact that
  // we're using a linear activation function.
  dydw1(0, 0) =
      nn.weights.at(0).row(0).dot(input) + nn.biases.at(0)(0);  // dy/da_00^1
  dydw1(0, 1) =
      nn.weights.at(0).row(1).dot(input) + nn.biases.at(0)(1);  // dy/da_01^1
  dydw1(0, 2) =
      nn.weights.at(0).row(2).dot(input) + nn.biases.at(0)(2);  // dy/da_02^1

  // Vectorized computation
  dydw1_vectorized = (nn.weights.at(0) * input + nn.biases.at(0)).transpose();

  std::cerr << "Numerical Gradient: " << std::endl
            << dydw1_numerical << std::endl;
  std::cerr << "Analytical Gradient: " << std::endl << dydw1 << std::endl;
  std::cerr << "Vectorized Gradient: " << std::endl
            << dydw1_vectorized << std::endl;
  std::cerr << "Difference of Gradients: " << std::endl
            << dydw1 - dydw1_numerical << std::endl;
}

void ComputeGradientLayerZeroWeightsSigmoid(const NeuralNetworkParameters& nn,
                                            const Eigen::VectorXd& input) {
  const size_t layer = 0;
  const Eigen::MatrixXd& w0 = nn.weights.at(layer);
  // std::cerr << "Layer Zero Input: " << input.transpose() << std::endl;
  // std::cerr << "Layer Zero Weights 0: " << std::endl << w0 << std::endl;

  // Compute network output for the original network.
  Eigen::VectorXd output;
  EvaluateNetwork(input, nn, &output);

  // Gradient matrix to be filled in.
  // Eigen::MatrixXd dydw0 = Eigen::MatrixXd::Zero(w0.rows(), w0.cols());
  Eigen::MatrixXd dydw0_numerical = Eigen::MatrixXd::Zero(w0.rows(), w0.cols());
  Eigen::MatrixXd dydw0_vectorized =
      Eigen::MatrixXd::Zero(w0.rows(), w0.cols());

  const double dw = 1e-6;
  for (size_t i = 0; i < w0.rows(); ++i) {
    for (size_t j = 0; j < w0.cols(); ++j) {
      // Perturb this particular element of the weight matrix.
      NeuralNetworkParameters nn_perturbation = nn;
      nn_perturbation.weights.at(layer)(i, j) += dw;

      // Compute network output for the perturbed network.
      Eigen::VectorXd output_perturbation;
      EvaluateNetwork(input, nn_perturbation, &output_perturbation);

      // Compute scalar value of numerical gradient.
      const Eigen::VectorXd vector_grad = (output_perturbation - output) / dw;
      assert(vector_grad.size() == 1);
      const double scalar_grad = vector_grad(0);
      dydw0_numerical(i, j) = scalar_grad;
    }
  }

  const Eigen::VectorXd l0_pre_act = nn.weights.at(0) * input + nn.biases.at(0);

  Eigen::VectorXd l0_post_act_grad;
  const Eigen::VectorXd l0_post_act =
      Activation(l0_pre_act, ActivationFunction::SIGMOID, &l0_post_act_grad);

  const Eigen::VectorXd l1_pre_act =
      nn.weights.at(1) * l0_post_act + nn.biases.at(1);

  Eigen::VectorXd l1_post_act_grad;
  const Eigen::VectorXd l1_post_act =
      Activation(l1_pre_act, ActivationFunction::SIGMOID, &l1_post_act_grad);

  // Vectorized computation
  // dydw0_vectorized = l1_post_act_grad.transpose() * nn.weights.at(1) *
  //                    l0_post_act_grad.transpose() * input;

  // THE KEY IS THAT WHEN YOU MULTIPLY IN THE GRADIENTS, IT'S ELEMENTWISE!
  Eigen::VectorXd foo(3);
  foo(0) = nn.weights.at(1)(0, 0);
  foo(1) = nn.weights.at(1)(0, 1);
  foo(2) = nn.weights.at(1)(0, 2);
  const double bar = l1_post_act_grad(0);
  dydw0_vectorized =
      bar * foo.cwiseProduct(l0_post_act_grad) * input.transpose();

  // std::cerr << "l1 post act grad: " << std::endl
  //           << l1_post_act_grad << std::endl;
  // std::cerr << "w1: " << std::endl << nn.weights.at(1) << std::endl;
  // std::cerr << "l0 post act grad: " << std::endl
  //           << l0_post_act_grad << std::endl;
  // std::cerr << "input: " << std::endl << input << std::endl;
  // std::cerr << "result_so_far: " << std::endl << result_so_far << std::endl;

  std::cerr << "Numerical Gradient: " << std::endl
            << dydw0_numerical << std::endl;
  std::cerr << "Vectorized Gradient: " << std::endl
            << dydw0_vectorized << std::endl;
  std::cerr << "Difference of Gradients: " << std::endl
            << dydw0_vectorized - dydw0_numerical << std::endl;
}

void ComputeGradientLayerOneWeightsSigmoid(const NeuralNetworkParameters& nn,
                                           const Eigen::VectorXd& input) {
  const size_t layer = 1;
  const Eigen::MatrixXd& w1 = nn.weights.at(layer);
  // std::cerr << "Layer Zero Input: " << input.transpose() << std::endl;
  // std::cerr << "Layer Zero Weights 0: " << std::endl << w1 << std::endl;

  // Compute network output for the original network.
  Eigen::VectorXd output;
  EvaluateNetwork(input, nn, &output);

  // Gradient matrix to be filled in.
  // Eigen::MatrixXd dydw1 = Eigen::MatrixXd::Zero(w1.rows(), w1.cols());
  Eigen::MatrixXd dydw1_numerical = Eigen::MatrixXd::Zero(w1.rows(), w1.cols());
  Eigen::MatrixXd dydw1_vectorized =
      Eigen::MatrixXd::Zero(w1.rows(), w1.cols());

  const double dw = 1e-6;
  for (size_t i = 0; i < w1.rows(); ++i) {
    for (size_t j = 0; j < w1.cols(); ++j) {
      // Perturb this particular element of the weight matrix.
      NeuralNetworkParameters nn_perturbation = nn;
      nn_perturbation.weights.at(layer)(i, j) += dw;

      // Compute network output for the perturbed network.
      Eigen::VectorXd output_perturbation;
      EvaluateNetwork(input, nn_perturbation, &output_perturbation);

      // Compute scalar value of numerical gradient.
      const Eigen::VectorXd vector_grad = (output_perturbation - output) / dw;
      assert(vector_grad.size() == 1);
      const double scalar_grad = vector_grad(0);
      dydw1_numerical(i, j) = scalar_grad;
    }
  }

  const Eigen::VectorXd l0_pre_act = nn.weights.at(0) * input + nn.biases.at(0);

  Eigen::VectorXd l0_post_act_grad;
  const Eigen::VectorXd l0_post_act =
      Activation(l0_pre_act, ActivationFunction::SIGMOID, &l0_post_act_grad);

  const Eigen::VectorXd l1_pre_act =
      nn.weights.at(1) * l0_post_act + nn.biases.at(1);

  Eigen::VectorXd l1_post_act_grad;
  const Eigen::VectorXd l1_post_act =
      Activation(l1_pre_act, ActivationFunction::SIGMOID, &l1_post_act_grad);

  // Vectorized computation
  dydw1_vectorized = l1_post_act_grad.transpose() * l0_post_act.transpose();

  std::cerr << "Numerical Gradient: " << std::endl
            << dydw1_numerical << std::endl;
  std::cerr << "Vectorized Gradient: " << std::endl
            << dydw1_vectorized << std::endl;
  std::cerr << "Difference of Gradients: " << std::endl
            << dydw1_vectorized - dydw1_numerical << std::endl;
}

int main() {
  // Randomly generate a neural network
  const int input_dimension = 2;
  const int output_dimension = 1;
  const int num_hidden_layers = 1;
  const int nodes_per_hidden_layer = 3;
  const Eigen::VectorXd input = Eigen::VectorXd::Random(input_dimension);

  // NeuralNetworkParameters nn = GetRandomNeuralNetwork(
  //     input_dimension, output_dimension, num_hidden_layers,
  //     nodes_per_hidden_layer, ActivationFunction::LINEAR,
  //     ActivationFunction::LINEAR);
  // ComputeGradientLayerZeroWeightsLinear(nn, input);
  // ComputeGradientLayerOneWeightsLinear(nn, input);

  NeuralNetworkParameters nn = GetRandomNeuralNetwork(
      input_dimension, output_dimension, num_hidden_layers,
      nodes_per_hidden_layer, ActivationFunction::SIGMOID,
      ActivationFunction::SIGMOID);
  ComputeGradientLayerZeroWeightsSigmoid(nn, input);
  ComputeGradientLayerOneWeightsSigmoid(nn, input);

  return 0;
}

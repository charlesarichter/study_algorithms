#include <iostream>

#include "nn.hpp"

void ComputeGradientTest(const NeuralNetworkParameters& nn,
                         const Eigen::VectorXd& input) {
  const size_t num_layers = nn.weights.size();
  for (size_t layer = 0; layer < num_layers; ++layer) {
    const Eigen::MatrixXd& w = nn.weights.at(layer);

    // Compute network output for the original network.
    Eigen::VectorXd output;
    std::vector<Eigen::MatrixXd> weight_gradients;
    EvaluateNetwork(input, nn, &output, &weight_gradients);

    // Gradient matrix to be filled in.
    Eigen::MatrixXd dydw_numerical = Eigen::MatrixXd::Zero(w.rows(), w.cols());
    Eigen::MatrixXd dydw_vectorized = Eigen::MatrixXd::Zero(w.rows(), w.cols());

    const double dw = 1e-3;
    for (size_t i = 0; i < w.rows(); ++i) {
      for (size_t j = 0; j < w.cols(); ++j) {
        // Perturb this particular element of the weight matrix.
        NeuralNetworkParameters nn_perturbation_plus = nn;
        nn_perturbation_plus.weights.at(layer)(i, j) += dw;

        // Compute network output for the perturbed network.
        Eigen::VectorXd output_perturbation_plus;
        std::vector<Eigen::MatrixXd> weight_gradients_perturbation_plus;
        EvaluateNetwork(input, nn_perturbation_plus, &output_perturbation_plus,
                        &weight_gradients_perturbation_plus);

        // Perturb this particular element of the weight matrix.
        NeuralNetworkParameters nn_perturbation_minus = nn;
        nn_perturbation_minus.weights.at(layer)(i, j) -= dw;

        // Compute network output for the perturbed network.
        Eigen::VectorXd output_perturbation_minus;
        std::vector<Eigen::MatrixXd> weight_gradients_perturbation_minus;
        EvaluateNetwork(input, nn_perturbation_minus,
                        &output_perturbation_minus,
                        &weight_gradients_perturbation_minus);

        // Compute scalar value of numerical gradient.
        const Eigen::VectorXd vector_grad =
            (output_perturbation_plus - output_perturbation_minus) / (2 * dw);
        assert(vector_grad.size() == 1);
        const double scalar_grad = vector_grad(0);
        dydw_numerical(i, j) = scalar_grad;
      }
    }
    std::cerr << "derivative of layer " << layer << " weights:" << std::endl
              << dydw_numerical << std::endl;
  }

  // dy/dA3 = f3'(layer_3_pre_act) * layer_2_post_act
  // dy/dA2 = f3'(layer_3_pre_act) * A3
  //        * f2'(layer_2_pre_act) * layer_1_post_act
  // dy/dA1 = f3'(layer_3_pre_act) * A3
  //        * f2'(layer_2_pre_act) * A2
  //        * f1'(layer_1_pre_act) * layer_0_post_act
  // dy/dA0 = f3'(layer_3_pre_act) * A3
  //        * f2'(layer_2_pre_act) * A2
  //        * f1'(layer_1_pre_act) * A1
  //        * f0'(layer_0_pre_act) * input

  const Eigen::VectorXd l0_pre_act = nn.weights.at(0) * input + nn.biases.at(0);

  Eigen::VectorXd l0_post_act_grad;
  const Eigen::VectorXd l0_post_act =
      Activation(l0_pre_act, ActivationFunction::SIGMOID, &l0_post_act_grad);

  const Eigen::VectorXd l1_pre_act =
      nn.weights.at(1) * l0_post_act + nn.biases.at(1);

  Eigen::VectorXd l1_post_act_grad;
  const Eigen::VectorXd l1_post_act =
      Activation(l1_pre_act, ActivationFunction::SIGMOID, &l1_post_act_grad);

  const Eigen::VectorXd l2_pre_act =
      nn.weights.at(2) * l1_post_act + nn.biases.at(2);

  Eigen::VectorXd l2_post_act_grad;
  const Eigen::VectorXd l2_post_act =
      Activation(l2_pre_act, ActivationFunction::SIGMOID, &l2_post_act_grad);

  const Eigen::VectorXd l3_pre_act =
      nn.weights.at(3) * l2_post_act + nn.biases.at(3);

  Eigen::VectorXd l3_post_act_grad;
  const Eigen::VectorXd l3_post_act =
      Activation(l3_pre_act, ActivationFunction::SIGMOID, &l3_post_act_grad);

  // dy/dA3 = f3'(layer_2_pre_act) * layer_2_post_act
  const Eigen::MatrixXd z = Eigen::MatrixXd::Ones(1, 1);
  const Eigen::MatrixXd dydw3 =
      z.cwiseProduct(l3_post_act_grad.transpose()).transpose() *
      l2_post_act.transpose();
  std::cerr << "dydw3 " << std::endl << dydw3 << std::endl;

  // dy/dA2 = f3'(layer_3_pre_act) * A3
  //         * f2'(layer_2_pre_act) * layer_1_post_act
  const Eigen::MatrixXd a =
      z.cwiseProduct(l3_post_act_grad.transpose()) * nn.weights.at(3);
  const Eigen::MatrixXd dydw2 =
      a.cwiseProduct(l2_post_act_grad.transpose()).transpose() *
      l1_post_act.transpose();
  std::cerr << "dydw2 " << std::endl << dydw2 << std::endl;

  // dy/dA1 = f3'(layer_3_pre_act) * A3
  //        * f2'(layer_2_pre_act) * A2
  //        * f1'(layer_1_pre_act) * layer_0_post_act
  const Eigen::MatrixXd b =
      a.cwiseProduct(l2_post_act_grad.transpose()) * nn.weights.at(2);
  const Eigen::MatrixXd dydw1 =
      b.cwiseProduct(l1_post_act_grad.transpose()).transpose() *
      l0_post_act.transpose();
  std::cerr << "dydw1 " << std::endl << dydw1 << std::endl;

  // dy/dA0 = f3'(layer_3_pre_act) * A3
  //        * f2'(layer_2_pre_act) * A2
  //        * f1'(layer_1_pre_act) * A1
  //        * f0'(layer_0_pre_act) * input
  const Eigen::MatrixXd c =
      b.cwiseProduct(l1_post_act_grad.transpose()) * nn.weights.at(1);
  const Eigen::MatrixXd dydw0 =
      c.cwiseProduct(l0_post_act_grad.transpose()).transpose() *
      input.transpose();
  std::cerr << "dydw0 " << std::endl << dydw0 << std::endl;
}

int main() {
  const int input_dimension = 2;
  const int output_dimension = 1;
  const int num_hidden_layers = 3;
  const int nodes_per_hidden_layer = 3;
  const Eigen::VectorXd input = Eigen::VectorXd::Random(input_dimension);
  NeuralNetworkParameters nn = GetRandomNeuralNetwork(
      input_dimension, output_dimension, num_hidden_layers,
      nodes_per_hidden_layer, ActivationFunction::SIGMOID,
      ActivationFunction::SIGMOID);
  ComputeGradientTest(nn, input);

  return 0;
}

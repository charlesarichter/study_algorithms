#include <iostream>

#include "nn.hpp"

void ComputeNetworkGradientsNumerically(
    const Eigen::VectorXd& input, const NeuralNetworkParameters& nn,
    std::vector<Eigen::MatrixXd>* numerical_weight_gradients,
    std::vector<Eigen::VectorXd>* numerical_bias_gradients) {
  const size_t num_layers = nn.weights.size();
  const double delta = 1e-6;
  for (size_t layer = 0; layer < num_layers; ++layer) {
    const Eigen::MatrixXd& w = nn.weights.at(layer);

    // Compute network output for the original network.
    Eigen::VectorXd output;
    std::vector<Eigen::MatrixXd> weight_gradients;
    std::vector<Eigen::VectorXd> bias_gradients;
    EvaluateNetwork(input, nn, &output, &weight_gradients, &bias_gradients);

    // Gradient matrix to be filled in.
    Eigen::MatrixXd dydw_numerical = Eigen::MatrixXd::Zero(w.rows(), w.cols());
    Eigen::VectorXd dydb_numerical = Eigen::VectorXd::Zero(w.rows());

    for (size_t i = 0; i < w.rows(); ++i) {
      // Perturb this particular element of the bias vector.
      NeuralNetworkParameters nn_perturbation_bias_plus = nn;
      nn_perturbation_bias_plus.biases.at(layer)(i) += delta;

      // Compute network output for the perturbed network.
      Eigen::VectorXd output_perturbation_bias_plus;
      std::vector<Eigen::MatrixXd> weight_gradients_perturbation_bias_plus;
      std::vector<Eigen::VectorXd> bias_gradients_perturbation_bias_plus;
      EvaluateNetwork(input, nn_perturbation_bias_plus,
                      &output_perturbation_bias_plus,
                      &weight_gradients_perturbation_bias_plus,
                      &bias_gradients_perturbation_bias_plus);

      // Perturb this particular element of the bias vector.
      NeuralNetworkParameters nn_perturbation_bias_minus = nn;
      nn_perturbation_bias_minus.biases.at(layer)(i) -= delta;

      // Compute network output for the perturbed network.
      Eigen::VectorXd output_perturbation_bias_minus;
      std::vector<Eigen::MatrixXd> weight_gradients_perturbation_bias_minus;
      std::vector<Eigen::VectorXd> bias_gradients_perturbation_bias_minus;
      EvaluateNetwork(input, nn_perturbation_bias_minus,
                      &output_perturbation_bias_minus,
                      &weight_gradients_perturbation_bias_minus,
                      &bias_gradients_perturbation_bias_minus);

      // Compute scalar value of numerical gradient.
      const Eigen::VectorXd bias_vector_grad =
          (output_perturbation_bias_plus - output_perturbation_bias_minus) /
          (2 * delta);
      assert(bias_vector_grad.size() == 1);
      const double bias_scalar_grad = bias_vector_grad(0);
      dydb_numerical(i) = bias_scalar_grad;

      for (size_t j = 0; j < w.cols(); ++j) {
        // Perturb this particular element of the weight matrix.
        NeuralNetworkParameters nn_perturbation_plus = nn;
        nn_perturbation_plus.weights.at(layer)(i, j) += delta;

        // Compute network output for the perturbed network.
        Eigen::VectorXd output_perturbation_plus;
        std::vector<Eigen::MatrixXd> weight_gradients_perturbation_plus;
        std::vector<Eigen::VectorXd> bias_gradients_perturbation_plus;
        EvaluateNetwork(input, nn_perturbation_plus, &output_perturbation_plus,
                        &weight_gradients_perturbation_plus,
                        &bias_gradients_perturbation_plus);

        // Perturb this particular element of the weight matrix.
        NeuralNetworkParameters nn_perturbation_minus = nn;
        nn_perturbation_minus.weights.at(layer)(i, j) -= delta;

        // Compute network output for the perturbed network.
        Eigen::VectorXd output_perturbation_minus;
        std::vector<Eigen::MatrixXd> weight_gradients_perturbation_minus;
        std::vector<Eigen::VectorXd> bias_gradients_perturbation_minus;
        EvaluateNetwork(input, nn_perturbation_minus,
                        &output_perturbation_minus,
                        &weight_gradients_perturbation_minus,
                        &bias_gradients_perturbation_minus);

        // Compute scalar value of numerical gradient.
        const Eigen::VectorXd vector_grad =
            (output_perturbation_plus - output_perturbation_minus) /
            (2 * delta);
        assert(vector_grad.size() == 1);
        const double scalar_grad = vector_grad(0);
        dydw_numerical(i, j) = scalar_grad;
      }
    }

    // std::cerr << "derivative of layer " << layer << " weights:" << std::endl
    //           << dydw_numerical << std::endl;
    // std::cerr << "derivative of layer " << layer << " biases:" << std::endl
    //           << dydb_numerical << std::endl;

    // Store the computed gradients.
    numerical_weight_gradients->emplace_back(dydw_numerical);
    numerical_bias_gradients->emplace_back(dydb_numerical);
  }
}

void ComputeLossGradientsNumerically(
    const Eigen::VectorXd& input, const NeuralNetworkParameters& nn,
    const Eigen::VectorXd& label, const LossFunction& loss_function,
    std::vector<Eigen::MatrixXd>* numerical_weight_gradients,
    std::vector<Eigen::VectorXd>* numerical_bias_gradients) {
  const size_t num_layers = nn.weights.size();
  const double delta = 1e-6;
  for (size_t layer = 0; layer < num_layers; ++layer) {
    const Eigen::MatrixXd& w = nn.weights.at(layer);

    // Compute network output for the original network.
    Eigen::VectorXd output;
    std::vector<Eigen::MatrixXd> weight_gradients;
    std::vector<Eigen::VectorXd> bias_gradients;
    EvaluateNetworkLoss(input, nn, label, loss_function, &output,
                        &weight_gradients, &bias_gradients);

    // Gradient matrix to be filled in.
    Eigen::MatrixXd dldw_numerical = Eigen::MatrixXd::Zero(w.rows(), w.cols());
    Eigen::VectorXd dldb_numerical = Eigen::VectorXd::Zero(w.rows());

    for (size_t i = 0; i < w.rows(); ++i) {
      // Perturb this particular element of the bias vector.
      NeuralNetworkParameters nn_perturbation_bias_plus = nn;
      nn_perturbation_bias_plus.biases.at(layer)(i) += delta;

      // Compute network output for the perturbed network.
      Eigen::VectorXd output_perturbation_bias_plus;
      std::vector<Eigen::MatrixXd> weight_gradients_perturbation_bias_plus;
      std::vector<Eigen::VectorXd> bias_gradients_perturbation_bias_plus;
      EvaluateNetworkLoss(input, nn_perturbation_bias_plus, label,
                          loss_function, &output_perturbation_bias_plus,
                          &weight_gradients_perturbation_bias_plus,
                          &bias_gradients_perturbation_bias_plus);

      // Perturb this particular element of the bias vector.
      NeuralNetworkParameters nn_perturbation_bias_minus = nn;
      nn_perturbation_bias_minus.biases.at(layer)(i) -= delta;

      // Compute network output for the perturbed network.
      Eigen::VectorXd output_perturbation_bias_minus;
      std::vector<Eigen::MatrixXd> weight_gradients_perturbation_bias_minus;
      std::vector<Eigen::VectorXd> bias_gradients_perturbation_bias_minus;
      EvaluateNetworkLoss(input, nn_perturbation_bias_minus, label,
                          loss_function, &output_perturbation_bias_minus,
                          &weight_gradients_perturbation_bias_minus,
                          &bias_gradients_perturbation_bias_minus);

      // Compute scalar value of numerical gradient.
      const Eigen::VectorXd bias_vector_grad =
          (output_perturbation_bias_plus - output_perturbation_bias_minus) /
          (2 * delta);
      assert(bias_vector_grad.size() == 1);
      const double bias_scalar_grad = bias_vector_grad(0);
      dldb_numerical(i) = bias_scalar_grad;

      for (size_t j = 0; j < w.cols(); ++j) {
        // Perturb this particular element of the weight matrix.
        NeuralNetworkParameters nn_perturbation_plus = nn;
        nn_perturbation_plus.weights.at(layer)(i, j) += delta;

        // Compute network output for the perturbed network.
        Eigen::VectorXd output_perturbation_plus;
        std::vector<Eigen::MatrixXd> weight_gradients_perturbation_plus;
        std::vector<Eigen::VectorXd> bias_gradients_perturbation_plus;
        EvaluateNetworkLoss(input, nn_perturbation_plus, label, loss_function,
                            &output_perturbation_plus,
                            &weight_gradients_perturbation_plus,
                            &bias_gradients_perturbation_plus);

        // Perturb this particular element of the weight matrix.
        NeuralNetworkParameters nn_perturbation_minus = nn;
        nn_perturbation_minus.weights.at(layer)(i, j) -= delta;

        // Compute network output for the perturbed network.
        Eigen::VectorXd output_perturbation_minus;
        std::vector<Eigen::MatrixXd> weight_gradients_perturbation_minus;
        std::vector<Eigen::VectorXd> bias_gradients_perturbation_minus;
        EvaluateNetworkLoss(input, nn_perturbation_minus, label, loss_function,
                            &output_perturbation_minus,
                            &weight_gradients_perturbation_minus,
                            &bias_gradients_perturbation_minus);

        // Compute scalar value of numerical gradient.
        const Eigen::VectorXd vector_grad =
            (output_perturbation_plus - output_perturbation_minus) /
            (2 * delta);
        assert(vector_grad.size() == 1);
        const double scalar_grad = vector_grad(0);
        dldw_numerical(i, j) = scalar_grad;
      }
    }

    // std::cerr << "derivative of layer " << layer << " weights:" << std::endl
    //           << dldw_numerical << std::endl;
    // std::cerr << "derivative of layer " << layer << " biases:" << std::endl
    //           << dldb_numerical << std::endl;

    // Store the computed gradients.
    numerical_weight_gradients->emplace_back(dldw_numerical);
    numerical_bias_gradients->emplace_back(dldb_numerical);
  }
}

void ComputeGradientsThreeHiddenLayerHardcoded(
    const Eigen::VectorXd& input, const NeuralNetworkParameters& nn,
    std::vector<Eigen::MatrixXd>* manual_weight_gradients,
    std::vector<Eigen::VectorXd>* manual_bias_gradients) {
  if (nn.weights.size() != 4) {
    throw std::runtime_error(
        "This function is hard-coded for a network with three hidden layers");
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

  // Manual.
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

  manual_weight_gradients->resize(nn.weights.size());
  manual_bias_gradients->resize(nn.weights.size());

  // dy/dA3 = f3'(layer_2_pre_act) * layer_2_post_act
  const Eigen::MatrixXd a = Eigen::MatrixXd::Ones(1, 1);
  const Eigen::MatrixXd dydw3 =
      a.cwiseProduct(l3_post_act_grad.transpose()).transpose() *
      l2_post_act.transpose();

  // dy/db3 = f3'(layer_2_pre_act)
  const Eigen::VectorXd dydb3 =
      a.cwiseProduct(l3_post_act_grad.transpose()).transpose();
  // std::cerr << "dydb3 " << std::endl << dydb3 << std::endl;

  manual_weight_gradients->at(3) = dydw3;
  manual_bias_gradients->at(3) = dydb3;

  // dy/dA2 = f3'(layer_3_pre_act) * A3
  //         * f2'(layer_2_pre_act) * layer_1_post_act
  const Eigen::MatrixXd b =
      a.cwiseProduct(l3_post_act_grad.transpose()) * nn.weights.at(3);
  const Eigen::MatrixXd dydw2 =
      b.cwiseProduct(l2_post_act_grad.transpose()).transpose() *
      l1_post_act.transpose();
  // std::cerr << "dydw2 " << std::endl << dydw2 << std::endl;

  // dy/db2 = f3'(layer_3_pre_act) * A3
  //         * f2'(layer_2_pre_act)
  const Eigen::VectorXd dydb2 =
      b.cwiseProduct(l2_post_act_grad.transpose()).transpose();
  // std::cerr << "dydb2 " << std::endl << dydb2 << std::endl;

  manual_weight_gradients->at(2) = dydw2;
  manual_bias_gradients->at(2) = dydb2;

  // dy/dA1 = f3'(layer_3_pre_act) * A3
  //        * f2'(layer_2_pre_act) * A2
  //        * f1'(layer_1_pre_act) * layer_0_post_act
  const Eigen::MatrixXd c =
      b.cwiseProduct(l2_post_act_grad.transpose()) * nn.weights.at(2);
  const Eigen::MatrixXd dydw1 =
      c.cwiseProduct(l1_post_act_grad.transpose()).transpose() *
      l0_post_act.transpose();
  // std::cerr << "dydw1 " << std::endl << dydw1 << std::endl;

  // dy/db1 = f3'(layer_3_pre_act) * A3
  //        * f2'(layer_2_pre_act) * A2
  //        * f1'(layer_1_pre_act)
  const Eigen::MatrixXd dydb1 =
      c.cwiseProduct(l1_post_act_grad.transpose()).transpose();
  // std::cerr << "dydb1 " << std::endl << dydb1 << std::endl;

  manual_weight_gradients->at(1) = dydw1;
  manual_bias_gradients->at(1) = dydb1;

  // dy/dA0 = f3'(layer_3_pre_act) * A3
  //        * f2'(layer_2_pre_act) * A2
  //        * f1'(layer_1_pre_act) * A1
  //        * f0'(layer_0_pre_act) * input
  const Eigen::MatrixXd d =
      c.cwiseProduct(l1_post_act_grad.transpose()) * nn.weights.at(1);
  const Eigen::MatrixXd dydw0 =
      d.cwiseProduct(l0_post_act_grad.transpose()).transpose() *
      input.transpose();
  // std::cerr << "dydw0 " << std::endl << dydw0 << std::endl;

  // dy/db0 = f3'(layer_3_pre_act) * A3
  //        * f2'(layer_2_pre_act) * A2
  //        * f1'(layer_1_pre_act) * A1
  //        * f0'(layer_0_pre_act)
  const Eigen::MatrixXd dydb0 =
      d.cwiseProduct(l0_post_act_grad.transpose()).transpose();
  // std::cerr << "dydb0 " << std::endl << dydb0 << std::endl;

  manual_weight_gradients->at(0) = dydw0;
  manual_bias_gradients->at(0) = dydb0;
}

void ComputeGradientsTest(const NeuralNetworkParameters& nn,
                          const Eigen::VectorXd& input) {
  // Containers for gradients comptued in different ways.
  std::vector<Eigen::MatrixXd> numerical_weight_gradients;
  std::vector<Eigen::VectorXd> numerical_bias_gradients;
  std::vector<Eigen::MatrixXd> backprop_weight_gradients;
  std::vector<Eigen::VectorXd> backprop_bias_gradients;

  // Backprop.
  Eigen::VectorXd backprop_output;
  EvaluateNetwork(input, nn, &backprop_output, &backprop_weight_gradients,
                  &backprop_bias_gradients);

  // Manual.
  ComputeNetworkGradientsNumerically(input, nn, &numerical_weight_gradients,
                                     &numerical_bias_gradients);

  // Compare gradients.
  for (int i = 0; i < backprop_weight_gradients.size(); ++i) {
    std::cerr << "Layer: " << i << std::endl;
    const Eigen::MatrixXd& dydw_backprop = backprop_weight_gradients.at(i);
    const Eigen::MatrixXd& dydw_numerical = numerical_weight_gradients.at(i);
    const Eigen::MatrixXd weight_gradient_difference =
        dydw_backprop - dydw_numerical;
    std::cerr << "Weight gradient difference: " << std::endl
              << weight_gradient_difference << std::endl;

    const Eigen::VectorXd& dydb_backprop = backprop_bias_gradients.at(i);
    const Eigen::VectorXd& dydb_numerical = numerical_bias_gradients.at(i);
    const Eigen::VectorXd bias_gradient_difference =
        dydb_backprop - dydb_numerical;
    std::cerr << "Bias gradient difference: " << std::endl
              << bias_gradient_difference << std::endl;
  }

  // Hardcoded 3 hidden layer
  std::vector<Eigen::MatrixXd> manual_weight_gradients;
  std::vector<Eigen::VectorXd> manual_bias_gradients;
  ComputeGradientsThreeHiddenLayerHardcoded(input, nn, &manual_weight_gradients,
                                            &manual_bias_gradients);

  // Compare gradients.
  for (int i = 0; i < manual_weight_gradients.size(); ++i) {
    std::cerr << "Layer: " << i << std::endl;
    const Eigen::MatrixXd& dydw_manual = manual_weight_gradients.at(i);
    const Eigen::MatrixXd& dydw_numerical = numerical_weight_gradients.at(i);
    const Eigen::MatrixXd weight_gradient_difference =
        dydw_manual - dydw_numerical;
    std::cerr << "Weight gradient difference: " << std::endl
              << weight_gradient_difference << std::endl;

    const Eigen::VectorXd& dydb_manual = manual_bias_gradients.at(i);
    const Eigen::VectorXd& dydb_numerical = numerical_bias_gradients.at(i);
    const Eigen::VectorXd bias_gradient_difference =
        dydb_manual - dydb_numerical;
    std::cerr << "Bias gradient difference: " << std::endl
              << bias_gradient_difference << std::endl;
  }
}

void ComputeLossTest(const NeuralNetworkParameters& nn,
                     const Eigen::VectorXd& input,
                     const Eigen::VectorXd& label) {
  Eigen::VectorXd net_output;
  std::vector<Eigen::MatrixXd> net_weight_gradients;
  std::vector<Eigen::VectorXd> net_bias_gradients;
  EvaluateNetwork(input, nn, &net_output, &net_weight_gradients,
                  &net_bias_gradients);

  // Just confirm we're getting a single output element.
  assert(net_output.size() == 1);

  // Hardcode the number of datapoints in this "batch" to be 1.
  const int num_data = 1;  // We would average over this many.

  // Cross Entropy / Log Loss function for a single data point.
  const double p_predicted = net_output(0);
  const double p_label = label(0);
  const double loss =
      -1 * (p_label * log(p_predicted) + (1 - p_label) * log(1 - p_predicted));

  std::cerr << "Manual calculation:" << std::endl;
  std::cerr << "Label: " << p_label << ", Prediction: " << p_predicted
            << ", Loss: " << loss << std::endl;

  Eigen::VectorXd loss_eval_function;
  std::vector<Eigen::MatrixXd> weight_gradients_eval_function;
  std::vector<Eigen::VectorXd> bias_gradients_eval_function;
  EvaluateNetworkLoss(input, nn, label, LossFunction::CROSS_ENTROPY,
                      &loss_eval_function, &weight_gradients_eval_function,
                      &bias_gradients_eval_function);
  std::cerr << "Loss computed by EvaluateNetworkLoss: " << std::endl
            << loss_eval_function << std::endl;

  // Want: dloss/dweights
  // Have: dloss/dppredicted, dppredicted/weights
  // dloss/dweights = dloss/dpredicted * dpredicted/dweights

  // "log" = natural logarithm, i.e. "ln". Does not hold for other bases.
  // Using the fact that d/dx(log(x)) = 1/x:

  // loss = -p_label * log(p_predicted) - (1 - p_label) * log(1 - p_predicted)
  // dloss/dpredicted = -p_label * d/dpredicted(log(p_predicted))
  //                    + (p_label - 1) * d/dpredicted(log(1 - p_predicted))
  //                  = -p_label * (1/p_predicted)
  //                    + (p_label - 1) * -1 * 1/(1 - p_predicted)

  const double dloss_dpredicted =
      -p_label * (1 / p_predicted) - (p_label - 1) / (1 - p_predicted);

  // Compute gradients of loss w.r.t. weights numerically to compare.
  std::vector<Eigen::MatrixXd> loss_weight_gradients;
  std::vector<Eigen::VectorXd> loss_bias_gradients;
  ComputeLossGradientsNumerically(input, nn, label, LossFunction::CROSS_ENTROPY,
                                  &loss_weight_gradients, &loss_bias_gradients);

  // Start off with the last row of weights.
  for (size_t i = 0; i < 3; ++i) {
    const Eigen::MatrixXd analytical_weight_gradient =
        net_weight_gradients.at(i) * dloss_dpredicted;
    const Eigen::MatrixXd analytical_bias_gradient =
        net_bias_gradients.at(i) * dloss_dpredicted;
    // std::cerr << "Analytical gradient of layer " << i
    //           << " weights: " << std::endl
    //           << analytical_gradient << std::endl;
    // std::cerr << "Numerical gradient of layer " << i
    //           << " weights: " << std::endl
    //           << loss_weight_gradients.at(i) << std::endl;
    std::cerr << "Layer "
              << " (numerical - analytical) weight gradient difference: "
              << std::endl
              << analytical_weight_gradient - loss_weight_gradients.at(i)
              << std::endl;
    std::cerr << "Layer "
              << " (numerical - analytical) bias gradient difference: "
              << std::endl
              << analytical_bias_gradient - loss_bias_gradients.at(i)
              << std::endl;
  }

  // Compare gradients computed in EvaluateNetworkLoss with numerical gradients.
  for (size_t i = 0; i < 3; ++i) {
    const Eigen::MatrixXd analytical_weight_gradient =
        weight_gradients_eval_function.at(i);
    const Eigen::MatrixXd analytical_bias_gradient =
        bias_gradients_eval_function.at(i);
    std::cerr << "Layer "
              << " (numerical - analytical from EvaluateNetworkLoss) weight "
                 "gradient difference: "
              << std::endl
              << analytical_weight_gradient - loss_weight_gradients.at(i)
              << std::endl;
    std::cerr << "Layer "
              << " (numerical - analytical from EvaluateNetworkLoss) bias "
                 "gradient difference: "
              << std::endl
              << analytical_bias_gradient - loss_bias_gradients.at(i)
              << std::endl;
  }
}

// f(input; weights & biases) = output
// cost(intput ; weights & biases) = ||output - label|| = (1/2)*(output -
// label)^2. dcostdw = dcostdoutput * doutputdw
//         = 2(output - label)*doutputdw
//
// Cross-entropy loss for classification/binary labels.
//

void TrainBackpropTest(const NeuralNetworkParameters& nn,
                       const Eigen::VectorXd& input,
                       const Eigen::VectorXd& label) {
  const double step_size = 1e-2;
  const int max_iterations = 1000;
  NeuralNetworkParameters nn_update = nn;

  for (int i = 0; i < max_iterations; ++i) {
    Eigen::VectorXd loss;
    std::vector<Eigen::MatrixXd> weight_gradients;
    std::vector<Eigen::VectorXd> bias_gradients;
    EvaluateNetworkLoss(input, nn_update, label, LossFunction::CROSS_ENTROPY,
                        &loss, &weight_gradients, &bias_gradients);

    std::cerr << "Loss: " << loss << std::endl;

    // Take a step in the gradient direction.
    for (size_t j = 0; j < weight_gradients.size(); ++j) {
      nn_update.weights.at(j) += -1 * step_size * weight_gradients.at(j);
    }
    for (size_t j = 0; j < bias_gradients.size(); ++j) {
      nn_update.biases.at(j) += -1 * step_size * bias_gradients.at(j);
    }
  }
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

  // Label for a single test datapoint.
  const Eigen::VectorXd label = Eigen::VectorXd::Ones(output_dimension);

  // ComputeGradientsTest(nn, input);
  // ComputeLossTest(nn, input, label);

  TrainBackpropTest(nn, input, label);

  return 0;
}

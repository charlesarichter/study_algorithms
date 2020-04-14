#include <fstream>
#include <iostream>
#include <string>

#include "conv.hpp"
#include "nn.hpp"
#include "training.hpp"

void ComputeNetworkGradientsNumerically(
    const Eigen::VectorXd& input, const NeuralNetworkParameters& nn,
    std::vector<std::vector<Eigen::MatrixXd>>* numerical_weight_gradients,
    std::vector<std::vector<Eigen::VectorXd>>* numerical_bias_gradients) {
  const size_t num_layers = nn.weights.size();
  const size_t output_dimension = nn.weights.back().rows();

  numerical_weight_gradients->resize(nn.weights.size());
  numerical_bias_gradients->resize(nn.weights.size());
  for (int i = 0; i < nn.weights.size(); ++i) {
    numerical_weight_gradients->at(i).resize(output_dimension);
    numerical_bias_gradients->at(i).resize(output_dimension);
  }

  const double delta = 1e-6;
  for (size_t k = 0; k < output_dimension; ++k) {
    for (size_t layer = 0; layer < num_layers; ++layer) {
      const Eigen::MatrixXd& w = nn.weights.at(layer);

      // // Compute network output for the original network.
      // Eigen::VectorXd output;
      // std::vector<std::vector<Eigen::MatrixXd>> weight_gradients;
      // std::vector<std::vector<Eigen::VectorXd>> bias_gradients;
      // EvaluateNetwork(input, nn, &output, &weight_gradients,
      // &bias_gradients);

      // Gradient matrix to be filled in.
      Eigen::MatrixXd dydw_numerical =
          Eigen::MatrixXd::Zero(w.rows(), w.cols());
      Eigen::VectorXd dydb_numerical = Eigen::VectorXd::Zero(w.rows());

      for (size_t i = 0; i < w.rows(); ++i) {
        // Perturb this particular element of the bias vector.
        NeuralNetworkParameters nn_perturbation_bias_plus = nn;
        nn_perturbation_bias_plus.biases.at(layer)(i) += delta;

        // Compute network output for the perturbed network.
        Eigen::VectorXd output_perturbation_bias_plus;
        std::vector<std::vector<Eigen::MatrixXd>>
            weight_gradients_perturbation_bias_plus;
        std::vector<std::vector<Eigen::VectorXd>>
            bias_gradients_perturbation_bias_plus;
        EvaluateNetwork(input, nn_perturbation_bias_plus,
                        &output_perturbation_bias_plus,
                        &weight_gradients_perturbation_bias_plus,
                        &bias_gradients_perturbation_bias_plus);

        // Perturb this particular element of the bias vector.
        NeuralNetworkParameters nn_perturbation_bias_minus = nn;
        nn_perturbation_bias_minus.biases.at(layer)(i) -= delta;

        // Compute network output for the perturbed network.
        Eigen::VectorXd output_perturbation_bias_minus;
        std::vector<std::vector<Eigen::MatrixXd>>
            weight_gradients_perturbation_bias_minus;
        std::vector<std::vector<Eigen::VectorXd>>
            bias_gradients_perturbation_bias_minus;
        EvaluateNetwork(input, nn_perturbation_bias_minus,
                        &output_perturbation_bias_minus,
                        &weight_gradients_perturbation_bias_minus,
                        &bias_gradients_perturbation_bias_minus);

        // Compute scalar value of numerical gradient.
        const Eigen::VectorXd bias_vector_grad =
            (output_perturbation_bias_plus - output_perturbation_bias_minus) /
            (2 * delta);
        assert(bias_vector_grad.size() == output_dimension);
        const double bias_scalar_grad = bias_vector_grad(k);  //(0);
        dydb_numerical(i) = bias_scalar_grad;

        for (size_t j = 0; j < w.cols(); ++j) {
          // Perturb this particular element of the weight matrix.
          NeuralNetworkParameters nn_perturbation_plus = nn;
          nn_perturbation_plus.weights.at(layer)(i, j) += delta;

          // Compute network output for the perturbed network.
          Eigen::VectorXd output_perturbation_plus;
          std::vector<std::vector<Eigen::MatrixXd>>
              weight_gradients_perturbation_plus;
          std::vector<std::vector<Eigen::VectorXd>>
              bias_gradients_perturbation_plus;
          EvaluateNetwork(input, nn_perturbation_plus,
                          &output_perturbation_plus,
                          &weight_gradients_perturbation_plus,
                          &bias_gradients_perturbation_plus);

          // Perturb this particular element of the weight matrix.
          NeuralNetworkParameters nn_perturbation_minus = nn;
          nn_perturbation_minus.weights.at(layer)(i, j) -= delta;

          // Compute network output for the perturbed network.
          Eigen::VectorXd output_perturbation_minus;
          std::vector<std::vector<Eigen::MatrixXd>>
              weight_gradients_perturbation_minus;
          std::vector<std::vector<Eigen::VectorXd>>
              bias_gradients_perturbation_minus;
          EvaluateNetwork(input, nn_perturbation_minus,
                          &output_perturbation_minus,
                          &weight_gradients_perturbation_minus,
                          &bias_gradients_perturbation_minus);

          // Compute scalar value of numerical gradient.
          const Eigen::VectorXd vector_grad =
              (output_perturbation_plus - output_perturbation_minus) /
              (2 * delta);
          assert(vector_grad.size() == output_dimension);
          const double scalar_grad = vector_grad(k);  //(0);
          dydw_numerical(i, j) = scalar_grad;
        }
      }

      // std::cerr << "derivative of layer " << layer << " weights:" <<
      // std::endl << dydw_numerical << std::endl;
      // std::cerr << "derivative of layer " << layer << " biases:" << std::endl
      //           << dydb_numerical << std::endl;

      // Store the computed gradients.
      numerical_weight_gradients->at(layer).at(k) = dydw_numerical;
      numerical_bias_gradients->at(layer).at(k) = dydb_numerical;
    }
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

    // std::cerr << "derivative of layer " << layer << " weights:" <<
    // std::endl
    //           << dldw_numerical << std::endl;
    // std::cerr << "derivative of layer " << layer << " biases:" <<
    // std::endl
    //           << dldb_numerical << std::endl;

    // Store the computed gradients.
    numerical_weight_gradients->emplace_back(dldw_numerical);
    numerical_bias_gradients->emplace_back(dldb_numerical);
  }
}

void ComputeGradientsThreeHiddenLayerHardcoded(
    const Eigen::VectorXd& input, const NeuralNetworkParameters& nn,
    std::vector<std::vector<Eigen::MatrixXd>>* manual_weight_gradients,
    std::vector<std::vector<Eigen::VectorXd>>* manual_bias_gradients) {
  // Check that we have a network with three hidden layers and one output layer.
  if (nn.weights.size() != 4 || nn.biases.size() != 4 ||
      nn.activation_functions.size() != 4) {
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
  Eigen::MatrixXd l0_post_act_grad;
  const Eigen::VectorXd l0_post_act =
      Activation(l0_pre_act, nn.activation_functions.at(0), &l0_post_act_grad);
  const Eigen::VectorXd l1_pre_act =
      nn.weights.at(1) * l0_post_act + nn.biases.at(1);
  Eigen::MatrixXd l1_post_act_grad;
  const Eigen::VectorXd l1_post_act =
      Activation(l1_pre_act, nn.activation_functions.at(1), &l1_post_act_grad);
  const Eigen::VectorXd l2_pre_act =
      nn.weights.at(2) * l1_post_act + nn.biases.at(2);
  Eigen::MatrixXd l2_post_act_grad;
  const Eigen::VectorXd l2_post_act =
      Activation(l2_pre_act, nn.activation_functions.at(2), &l2_post_act_grad);
  const Eigen::VectorXd l3_pre_act =
      nn.weights.at(3) * l2_post_act + nn.biases.at(3);
  Eigen::MatrixXd l3_post_act_grad;
  const Eigen::VectorXd l3_post_act =
      Activation(l3_pre_act, nn.activation_functions.at(3), &l3_post_act_grad);

  const int output_dimension = nn.weights.back().rows();

  manual_weight_gradients->resize(nn.weights.size());
  manual_bias_gradients->resize(nn.weights.size());
  for (int i = 0; i < nn.weights.size(); ++i) {
    manual_weight_gradients->at(i).resize(output_dimension);
    manual_bias_gradients->at(i).resize(output_dimension);
  }

  // Loop over the network outputs. Essentially we are structuring this
  // computation as if we have a separate network for each of the outputs. Use
  // this to find reused computation and condense/optimize if we can.
  for (int j = 0; j < output_dimension; ++j) {
    // dy/dA3 = f3'(layer_2_pre_act) * layer_2_post_act
    const Eigen::MatrixXd a = Eigen::MatrixXd::Ones(1, 1);
    const Eigen::MatrixXd dydw3 =
        (a * l3_post_act_grad.row(j)).transpose() * l2_post_act.transpose();

    // dy/db3 = f3'(layer_2_pre_act)
    const Eigen::VectorXd dydb3 = (a * l3_post_act_grad.row(j)).transpose();
    // std::cerr << "dydb3 " << std::endl << dydb3 << std::endl;

    manual_weight_gradients->at(3).at(j) = dydw3;
    manual_bias_gradients->at(3).at(j) = dydb3;

    // dy/dA2 = f3'(layer_3_pre_act) * A3
    //         * f2'(layer_2_pre_act) * layer_1_post_act
    const Eigen::MatrixXd b = a * l3_post_act_grad.row(j) * nn.weights.at(3);
    const Eigen::MatrixXd dydw2 =
        (b * l2_post_act_grad).transpose() * l1_post_act.transpose();
    // std::cerr << "dydw2 " << std::endl << dydw2 << std::endl;

    // dy/db2 = f3'(layer_3_pre_act) * A3
    //         * f2'(layer_2_pre_act)
    const Eigen::VectorXd dydb2 = (b * l2_post_act_grad).transpose();
    // std::cerr << "dydb2 " << std::endl << dydb2 << std::endl;

    manual_weight_gradients->at(2).at(j) = dydw2;
    manual_bias_gradients->at(2).at(j) = dydb2;

    // dy/dA1 = f3'(layer_3_pre_act) * A3
    //        * f2'(layer_2_pre_act) * A2
    //        * f1'(layer_1_pre_act) * layer_0_post_act
    const Eigen::MatrixXd c = b * l2_post_act_grad * nn.weights.at(2);
    const Eigen::MatrixXd dydw1 =
        (c * l1_post_act_grad).transpose() * l0_post_act.transpose();
    // std::cerr << "dydw1 " << std::endl << dydw1 << std::endl;

    // dy/db1 = f3'(layer_3_pre_act) * A3
    //        * f2'(layer_2_pre_act) * A2
    //        * f1'(layer_1_pre_act)
    const Eigen::MatrixXd dydb1 = (c * l1_post_act_grad).transpose();
    // std::cerr << "dydb1 " << std::endl << dydb1 << std::endl;

    manual_weight_gradients->at(1).at(j) = dydw1;
    manual_bias_gradients->at(1).at(j) = dydb1;

    // dy/dA0 = f3'(layer_3_pre_act) * A3
    //        * f2'(layer_2_pre_act) * A2
    //        * f1'(layer_1_pre_act) * A1
    //        * f0'(layer_0_pre_act) * input
    const Eigen::MatrixXd d = c * l1_post_act_grad * nn.weights.at(1);
    const Eigen::MatrixXd dydw0 =
        (d * l0_post_act_grad).transpose() * input.transpose();
    // std::cerr << "dydw0 " << std::endl << dydw0 << std::endl;

    // dy/db0 = f3'(layer_3_pre_act) * A3
    //        * f2'(layer_2_pre_act) * A2
    //        * f1'(layer_1_pre_act) * A1
    //        * f0'(layer_0_pre_act)
    const Eigen::MatrixXd dydb0 = (d * l0_post_act_grad).transpose();
    // std::cerr << "dydb0 " << std::endl << dydb0 << std::endl;

    manual_weight_gradients->at(0).at(j) = dydw0;
    manual_bias_gradients->at(0).at(j) = dydb0;
  }
}

void ComputeGradientsTest(const NeuralNetworkParameters& nn,
                          const Eigen::VectorXd& input) {
  // Containers for gradients comptued in different ways.
  std::vector<std::vector<Eigen::MatrixXd>> numerical_weight_gradients;
  std::vector<std::vector<Eigen::VectorXd>> numerical_bias_gradients;
  std::vector<std::vector<Eigen::MatrixXd>> backprop_weight_gradients;
  std::vector<std::vector<Eigen::VectorXd>> backprop_bias_gradients;

  // Backprop.
  Eigen::VectorXd backprop_output;
  EvaluateNetwork(input, nn, &backprop_output, &backprop_weight_gradients,
                  &backprop_bias_gradients);

  // Manual.
  ComputeNetworkGradientsNumerically(input, nn, &numerical_weight_gradients,
                                     &numerical_bias_gradients);

  // Compare gradients.
  for (int j = 0; j < backprop_weight_gradients.front().size(); ++j) {
    std::cerr << "Output: " << j << std::endl;
    for (int i = 0; i < backprop_weight_gradients.size(); ++i) {
      std::cerr << "Layer: " << i << std::endl;
      const Eigen::MatrixXd& dydw_backprop =
          backprop_weight_gradients.at(i).at(j);
      const Eigen::MatrixXd& dydw_numerical =
          numerical_weight_gradients.at(i).at(j);
      const Eigen::MatrixXd weight_gradient_difference =
          dydw_backprop - dydw_numerical;
      std::cerr << "Weight gradient difference: " << std::endl
                << weight_gradient_difference << std::endl;

      const Eigen::VectorXd& dydb_backprop =
          backprop_bias_gradients.at(i).at(j);
      const Eigen::VectorXd& dydb_numerical =
          numerical_bias_gradients.at(i).at(j);
      const Eigen::VectorXd bias_gradient_difference =
          dydb_backprop - dydb_numerical;
      std::cerr << "Bias gradient difference: " << std::endl
                << bias_gradient_difference << std::endl;
    }
  }

  // Hardcoded 3 hidden layer
  std::vector<std::vector<Eigen::MatrixXd>> manual_weight_gradients;
  std::vector<std::vector<Eigen::VectorXd>> manual_bias_gradients;
  ComputeGradientsThreeHiddenLayerHardcoded(input, nn, &manual_weight_gradients,
                                            &manual_bias_gradients);

  // Compare gradients.

  for (int j = 0; j < manual_weight_gradients.front().size(); ++j) {
    std::cerr << "Output: " << j << std::endl;
    for (int i = 0; i < manual_weight_gradients.size(); ++i) {
      std::cerr << "Layer: " << i << std::endl;
      const Eigen::MatrixXd& dydw_manual = manual_weight_gradients.at(i).at(j);
      const Eigen::MatrixXd& dydw_numerical =
          numerical_weight_gradients.at(i).at(j);
      const Eigen::MatrixXd weight_gradient_difference =
          dydw_manual - dydw_numerical;
      std::cerr << "Weight gradient difference: " << std::endl
                << weight_gradient_difference << std::endl;

      const Eigen::VectorXd& dydb_manual = manual_bias_gradients.at(i).at(j);
      const Eigen::VectorXd& dydb_numerical =
          numerical_bias_gradients.at(i).at(j);
      const Eigen::VectorXd bias_gradient_difference =
          dydb_manual - dydb_numerical;
      std::cerr << "Bias gradient difference: " << std::endl
                << bias_gradient_difference << std::endl;
    }
  }
}

void ComputeCrossEntropyLossTest(const NeuralNetworkParameters& nn,
                                 const Eigen::VectorXd& input,
                                 const Eigen::VectorXd& label) {
  const LossFunction loss_function = LossFunction::CROSS_ENTROPY;

  Eigen::VectorXd net_output;
  std::vector<std::vector<Eigen::MatrixXd>> net_weight_gradients;
  std::vector<std::vector<Eigen::VectorXd>> net_bias_gradients;
  EvaluateNetwork(input, nn, &net_output, &net_weight_gradients,
                  &net_bias_gradients);

  // Just confirm we're getting a single output element.
  // assert(net_output.size() == 1);

  // // Hardcode the number of datapoints in this "batch" to be 1.
  // const int num_data = 1;  // We would average over this many.
  //
  // // Binary Cross Entropy / Log Loss function for a single data point.
  // const double p_predicted = net_output(0);
  // const double p_label = label(0);
  // const double loss =
  //     -1 * (p_label * log(p_predicted) + (1 - p_label) * log(1 -
  //     p_predicted));
  //
  // Multi-class Cross Entropy / Log Loss function for a single data point.
  assert(label.size() == net_output.size());
  double loss = 0;
  Eigen::VectorXd dloss_dpredicted(net_output.size());
  for (size_t i = 0; i < label.size(); ++i) {
    loss += -1 * (label(i) * log(net_output(i)) +
                  (1 - label(i)) * log(1 - net_output(i)));
    dloss_dpredicted(i) =
        -label(i) / net_output(i) - (label(i) - 1) / (1 - net_output(i));
  }

  std::cerr << "Loss computed by manual calculation: " << loss << std::endl;
  // std::cerr << "Label: " << p_label << ", Prediction: " << p_predicted
  //           << ", Loss: " << loss << std::endl;

  Eigen::VectorXd loss_eval_function;
  std::vector<Eigen::MatrixXd> weight_gradients_eval_function;
  std::vector<Eigen::VectorXd> bias_gradients_eval_function;
  EvaluateNetworkLoss(input, nn, label, loss_function, &loss_eval_function,
                      &weight_gradients_eval_function,
                      &bias_gradients_eval_function);
  std::cerr << "Loss computed by EvaluateNetworkLoss: " << loss_eval_function
            << std::endl;

  Eigen::VectorXd loss_eval_function_combined;
  std::vector<Eigen::MatrixXd> weight_gradients_eval_function_combined;
  std::vector<Eigen::VectorXd> bias_gradients_eval_function_combined;
  EvaluateNetworkLossCombinedImplementation(
      input, nn, label, loss_function, &loss_eval_function_combined,
      &weight_gradients_eval_function_combined,
      &bias_gradients_eval_function_combined);
  std::cerr << "Loss computed by EvaluateNetworkLossCombinedImplementation: "
            << loss_eval_function_combined << std::endl;

  // // Want: dloss/dweights
  // // Have: dloss/dppredicted, dppredicted/weights
  // // dloss/dweights = dloss/dpredicted * dpredicted/dweights
  //
  // // "log" = natural logarithm, i.e. "ln". Does not hold for other bases.
  // // Using the fact that d/dx(log(x)) = 1/x:
  //
  // // loss = -p_label * log(p_predicted) - (1 - p_label) * log(1 -
  // // p_predicted)
  // // dloss/dpredicted = -p_label * d/dpredicted(log(p_predicted))
  // //                    + (p_label - 1) * d/dpredicted(log(1 - p_predicted))
  // //                  = -p_label * (1/p_predicted)
  // //                    + (p_label - 1) * -1 * 1/(1 - p_predicted)
  //
  // const double dloss_dpredicted =
  //     -p_label * (1 / p_predicted) - (p_label - 1) / (1 - p_predicted);

  // Compute gradients of loss w.r.t. weights numerically to compare.
  std::vector<Eigen::MatrixXd> loss_weight_gradients;
  std::vector<Eigen::VectorXd> loss_bias_gradients;
  ComputeLossGradientsNumerically(input, nn, label, loss_function,
                                  &loss_weight_gradients, &loss_bias_gradients);

  // Start off with the last row of weights.
  for (size_t i = 0; i < 3; ++i) {
    Eigen::MatrixXd analytical_weight_gradient =
        Eigen::MatrixXd::Zero(nn.weights.at(i).rows(), nn.weights.at(i).cols());
    Eigen::VectorXd analytical_bias_gradient =
        Eigen::VectorXd::Zero(nn.biases.at(i).size());
    for (size_t j = 0; j < net_output.size(); ++j) {
      analytical_weight_gradient +=
          net_weight_gradients.at(i).at(j) * dloss_dpredicted(j);
      analytical_bias_gradient +=
          net_bias_gradients.at(i).at(j) * dloss_dpredicted(j);
    }

    //   const Eigen::MatrixXd analytical_weight_gradient =
    //       net_weight_gradients.at(i) * dloss_dpredicted;
    //   const Eigen::MatrixXd analytical_bias_gradient =
    //       net_bias_gradients.at(i) * dloss_dpredicted;
    // std::cerr << "Analytical gradient of layer " << i
    //           << " weights: " << std::endl
    //           << analytical_weight_gradient << std::endl;
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

  // Compare gradients computed in EvaluateNetworkLossCombinedImplementation
  // with numerical gradients.
  for (size_t i = 0; i < 3; ++i) {
    const Eigen::MatrixXd analytical_weight_gradient =
        weight_gradients_eval_function_combined.at(i);
    const Eigen::MatrixXd analytical_bias_gradient =
        bias_gradients_eval_function_combined.at(i);
    std::cerr << "Layer "
              << " (numerical - analytical from "
                 "EvaluateNetworkLossCombinedImplementation) weight "
                 "gradient difference: "
              << std::endl
              << analytical_weight_gradient - loss_weight_gradients.at(i)
              << std::endl;
    std::cerr << "Layer "
              << " (numerical - analytical from "
                 "EvaluateNetworkLossCombinedImplementation) bias "
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
    EvaluateNetworkLossCombinedImplementation(
        input, nn_update, label, LossFunction::CROSS_ENTROPY, &loss,
        &weight_gradients, &bias_gradients);

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

void RunTests() {
  const int input_dimension = 2;
  const int output_dimension = 10;
  const int num_hidden_layers = 3;
  const int nodes_per_hidden_layer = 3;
  const Eigen::VectorXd input = Eigen::VectorXd::Random(input_dimension);
  NeuralNetworkParameters nn = GetRandomNeuralNetwork(
      input_dimension, output_dimension, num_hidden_layers,
      nodes_per_hidden_layer, ActivationFunction::RELU,
      ActivationFunction::SOFTMAX);

  // Label for a single test datapoint. One-hot with the first element = 1.
  Eigen::VectorXd label = Eigen::VectorXd::Zero(output_dimension);
  label(0) = 1;

  ComputeGradientsTest(nn, input);
  ComputeCrossEntropyLossTest(nn, input, label);
  TrainBackpropTest(nn, input, label);
}

void LoadMnist(const std::string& csv_filename, const size_t num_images,
               std::vector<Eigen::VectorXd>* images,
               std::vector<Eigen::VectorXd>* labels) {
  /**
   * See https://pjreddie.com/projects/mnist-in-csv/
   */
  const size_t num_classes = 10;
  const size_t num_pixels = 784;

  // Allocate data using fill constructor
  images->resize(num_images, Eigen::VectorXd(num_pixels));
  labels->resize(num_images, Eigen::VectorXd::Zero(num_classes));

  // File pointer
  std::fstream fin;

  // Open file
  fin.open(csv_filename, std::fstream::in);

  if (!fin.is_open()) {
    std::cerr << "File not open!" << std::endl;
  }

  // Read the Data from the file as String Vector
  size_t num_images_read = 0;
  std::string line;
  while (std::getline(fin, line)) {
    // used for breaking words
    std::stringstream s(line);

    // Read first element separately, since this is the class.
    std::string word;
    std::getline(s, word, ',');
    const int class_id = std::stoi(word);
    // std::cerr << "Class ID: " << class_id << std::endl;
    labels->at(num_images_read)[class_id] = 1.0;

    // Read pixels sequentially.
    Eigen::VectorXd& image = images->at(num_images_read);

    size_t pixel_index = 0;
    while (std::getline(s, word, ',')) {
      // For now, just scale to the interval (0,1) or (-1,1) right here.
      // TODO: Determine the best way to scale/center/normalize/transform the
      // data, possibly depending on which activation functions are being used.
      const double pixel_value =
          2 * static_cast<double>(std::stoi(word)) / 255 - 1;

      // Faster to pre-allocate and use [] than push_/emplace_back or .at()
      image[pixel_index] = pixel_value;
      ++pixel_index;
    }
    ++num_images_read;
  }
  return;
}

void MnistTest() {
  // Load training data.
  std::vector<Eigen::VectorXd> training_images;
  std::vector<Eigen::VectorXd> training_labels;
  LoadMnist("../data/mnist_train.csv", 60000, &training_images,
            &training_labels);

  // Load test data.
  std::vector<Eigen::VectorXd> test_images;
  std::vector<Eigen::VectorXd> test_labels;
  LoadMnist("../data/mnist_test.csv", 10000, &test_images, &test_labels);

  std::cerr << "Loaded data" << std::endl;

  // Specify network. TODO: Work on proper random weight initialization.
  const int input_dimension = training_images.at(0).size();
  const int output_dimension = training_labels.at(0).size();
  const int num_hidden_layers = 3;
  const int nodes_per_hidden_layer = 200;
  const Eigen::VectorXd input = Eigen::VectorXd::Random(input_dimension);
  NeuralNetworkParameters nn = GetRandomNeuralNetwork(
      input_dimension, output_dimension, num_hidden_layers,
      nodes_per_hidden_layer, ActivationFunction::RELU,
      ActivationFunction::SOFTMAX);
  // nodes_per_hidden_layer, ActivationFunction::SIGMOID,
  // ActivationFunction::SOFTMAX);
  std::cerr << "Generated initial network" << std::endl;

  // Training.
  // TrainBackpropTest(nn, images.at(0), labels.at(0));
  const NeuralNetworkParameters nn_trained =
      Train(nn, training_images, training_labels, test_images, test_labels);
}

int main() {
  // TODO: Read http://cs231n.github.io and in particular
  // http://cs231n.github.io/convolutional-networks/, which is a great resource
  // on neural nets and implementation details like how to formulate a
  // convolutional layer as one big matrix multiplication and a note on how the
  // backward pass is also a convolution with flipped filters. It has a good
  // explanation of the dimensionality of the input, output and filters.

  // RunTests();
  // MnistTest();
  // RunConvTests();
  // RunConvKernelTests();
  RunConvGradientTests();

  return 0;
}

// Notes for n-dimensional output:
// - Neural network output dimension can be arbitrary for output layers like
// softmax functions, which are used in n-class classification.
// - Even if the neural network output is n-dimensional, all n dimensions are
// combined into a single loss function, which means that the output of the
// loss is still one-dimensional.
// - For n-dimensional output, we compute the output of each element of the
// neural network with respect to each weight and bias variable. So to
// completely describe the weight and bias gradients, we would have:
// 1) n outputs
// 2) m weights (or biases) per layer
// 3) p layers
// So to capture the weight and bias gradients, instead of:
//    std::vector<Eigen::MatrixXd> weight_gradients;
//    std::vector<Eigen::VectorXd> bias_gradients;
// We would need:
//    std::vector<std::vector<Eigen::MatrixXd>> weight_gradients;
//    std::vector<std::vector<Eigen::VectorXd>> bias_gradients;
// Where the outer-most vector contains one element per layer, the
// second-outer-most vector contains one element per output channel, and the
// Eigen::MatrixXd (or Eigen::VectorXd) represents the gradient of the
// elements of the weights (or biases) in that layer.
//
// Next Task: Extend all calculations to handle n-dimensional output as
// described above, and test with softmax.
//
// Note that if we just care about gradients of weights/biases w.r.t. loss,
// then it may be overkill and inefficient to compute, store and return the
// gradients of weights/biases w.r.t. all of the individual outputs.
//
// The current code is pretty much entirely built around the assumption that
// the neural network output is one-dimensional. That is what allows us to
// use:
//    std::vector<Eigen::MatrixXd> weight_gradients;
//    std::vector<Eigen::VectorXd> bias_gradients;
// If we ever want the ability to compute gradients of some subset of the
// network, say the gradient of the output of the hidden layers (which may
// have n-dim output) or the pre-loss n-dim activation, we will need to extend
// our calculations to add a diension to the way we store and backpropagate
// gradients.

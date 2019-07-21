#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>

#include "training.hpp"

// NeuralNetworkParameters AdamOptimizer(
//     const NeuralNetworkParameters& current_params, const size_t t,
//     const std::vector<Eigen::MatrixXd>& weight_gradients,
//     const std::vector<Eigen::VectorXd>& bias_gradients) {
//   const double alpha = 0.001;
//   const double beta1 = 0.9;
//   const double beta2 = 0.999;
//   const double eps = 1e-8;
//
//   const double beta1t = pow(beta1, t);
//   const double beta2t = pow(beta2, t);
//   std::cerr << "Value of t: " << t << std::endl;
// }

void AdamOptimizer(const NeuralNetworkParameters& current_network_params,
                   const AdamOptimizerParameters& current_optimizer_params,
                   const std::vector<Eigen::MatrixXd>& weight_gradients,
                   const std::vector<Eigen::VectorXd>& bias_gradients,
                   const int iteration,
                   NeuralNetworkParameters* updated_network_params,
                   AdamOptimizerParameters* updated_optimizer_params) {
  // Copy over the current NN parameters (we will overwrite weights later).
  *updated_network_params = current_network_params;

  // Constants from the paper: https://arxiv.org/pdf/1412.6980.pdf
  double alpha = 0.001;
  double beta1 = 0.9;
  double beta2 = 0.999;
  double eps = 1e-8;
  int t = iteration + 1;

  // Require: α: Stepsize
  // Require: β1, β2 ∈ [0, 1): Exponential decay rates for the moment estimates
  // Require: f(θ): Stochastic objective function with parameters θ
  // Require: θ0: Initial parameter vector
  // m0 ← 0 (Initialize 1st moment vector)
  // v0 ← 0 (Initialize 2nd moment vector)
  // t ← 0 (Initialize timestep)
  // while θt not converged do
  //  t ← t + 1
  //  gt ← ∇θft(θt−1) (Get gradients w.r.t. stochastic objective at timestep t)
  //  mt ← β1 · mt−1 + (1 − β1) · gt (Update biased first moment estimate)
  //  vt ← β2 · vt−1 + (1 − β2) · gt^2 (Update biased second raw moment est)
  //  m^hat_t ← mt/(1 − β_1^t) (Compute bias-corrected first moment estimate)
  //  v^hat_t ← vt/(1 − β_2^t) (Compute bias-corrected second raw moment est)
  //  θt ← θt−1 − α · m^hat_t/(√v^hat_t + epsilon) (Update parameters)
  // end while
  // return θt(Resulting parameters)

  // std::vector<Eigen::MatrixXd> weight_grad_squared(weight_gradients.size());
  // std::vector<Eigen::VectorXd> bias_grad_squared(bias_gradients.size());
  // for (size_t i = 0; i < weight_gradients.size(); ++i) {
  //   weight_grad_squared.at(i) = weight_gradients.at(i).square();
  //   bias_grad_squared.at(i) = bias_gradients.at(i).square();
  // }

  // Allocate temporary storage for first and second moment estimates.
  std::vector<Eigen::MatrixXd> mtw(weight_gradients.size());
  std::vector<Eigen::VectorXd> mtb(bias_gradients.size());
  std::vector<Eigen::MatrixXd> vtw(weight_gradients.size());
  std::vector<Eigen::VectorXd> vtb(bias_gradients.size());

  // TODO: Don't need to store these.
  std::vector<Eigen::MatrixXd> mhat_tw(weight_gradients.size());
  std::vector<Eigen::VectorXd> mhat_tb(bias_gradients.size());
  std::vector<Eigen::MatrixXd> vhat_tw(weight_gradients.size());
  std::vector<Eigen::VectorXd> vhat_tb(bias_gradients.size());

  // Allocate temporary storage for updated weights.
  std::vector<Eigen::MatrixXd> weights_updated(weight_gradients.size());
  std::vector<Eigen::VectorXd> biases_updated(bias_gradients.size());

  // Loop over all layers in the network.
  for (size_t i = 0; i < weight_gradients.size(); ++i) {
    const Eigen::MatrixXd& gw = weight_gradients.at(i);
    const Eigen::VectorXd& gb = bias_gradients.at(i);

    // mt ← β1 · mt−1 + (1 − β1) · gt (Update biased first moment estimate)
    mtw.at(i) = beta1 * current_optimizer_params.weight_first_moment.at(i) +
                (1 - beta1) * weight_gradients.at(i);
    mtb.at(i) = beta1 * current_optimizer_params.bias_first_moment.at(i) +
                (1 - beta1) * bias_gradients.at(i);

    // vt ← β2 · vt−1 + (1 − β2) · gt^2 (Update biased second raw moment est)
    vtw.at(i) = beta2 * current_optimizer_params.weight_second_moment.at(i) +
                (1 - beta2) * gw.cwiseProduct(gw);
    vtb.at(i) = beta2 * current_optimizer_params.bias_second_moment.at(i) +
                (1 - beta2) * gb.cwiseProduct(gb);

    // m^hat_t ← mt/(1 − β_1^t) (Compute bias-corrected first moment estimate)
    mhat_tw.at(i) = mtw.at(i) / (1 - pow(beta1, t));
    mhat_tb.at(i) = mtb.at(i) / (1 - pow(beta1, t));

    // v^hat_t ← vt/(1 − β_2^t) (Compute bias-corrected second raw moment est)
    vhat_tw.at(i) = vtw.at(i) / (1 - pow(beta2, t));
    vhat_tb.at(i) = vtb.at(i) / (1 - pow(beta2, t));

    Eigen::MatrixXd ones_mat =
        Eigen::MatrixXd::Ones(current_network_params.weights.at(i).rows(),
                              current_network_params.weights.at(i).cols());
    Eigen::VectorXd ones_vec =
        Eigen::VectorXd::Ones(current_network_params.biases.at(i).size());

    // θt ← θt−1 − α · m^hat_t/(√v^hat_t + epsilon) (Update parameters)
    weights_updated.at(i) =
        current_network_params.weights.at(i) -
        alpha * mhat_tw.at(i).cwiseQuotient(vhat_tw.at(i).cwiseSqrt() +
                                            eps * ones_mat);
    biases_updated.at(i) =
        current_network_params.biases.at(i) -
        alpha * mhat_tb.at(i).cwiseQuotient(vhat_tb.at(i).cwiseSqrt() +
                                            eps * ones_vec);
  }

  // Copy updated weights to the output.
  // TODO: Eliminate unnecessary copies and memory allocation.
  updated_network_params->weights = weights_updated;
  updated_network_params->biases = biases_updated;
  updated_optimizer_params->weight_first_moment = mtw;
  updated_optimizer_params->bias_first_moment = mtb;
  updated_optimizer_params->weight_second_moment = vtw;
  updated_optimizer_params->bias_second_moment = vtb;
}

std::vector<size_t> GenerateRandomIndices(const size_t max_index,
                                          const size_t num_indices) {
  std::random_device rd;
  std::mt19937 g(rd());
  std::vector<size_t> index_sequence(max_index);
  std::iota(std::begin(index_sequence), std::end(index_sequence), 0);
  std::shuffle(index_sequence.begin(), index_sequence.end(), g);
  return std::vector<size_t>(index_sequence.begin(),
                             index_sequence.begin() + num_indices);
}

void ComputePerformanceOnTestSet(
    const NeuralNetworkParameters& nn,
    const std::vector<Eigen::VectorXd>& test_inputs,
    const std::vector<Eigen::VectorXd>& test_labels) {
  size_t num_correct = 0;
  size_t num_eval = 500;
  for (size_t i = 0; i < num_eval; ++i) {
    const Eigen::VectorXd& input = test_inputs.at(i);
    const Eigen::VectorXd& label = test_labels.at(i);
    Eigen::VectorXd output;
    std::vector<std::vector<Eigen::MatrixXd>> weight_gradients;
    std::vector<std::vector<Eigen::VectorXd>> bias_gradients;
    EvaluateNetwork(input, nn, &output, &weight_gradients, &bias_gradients);

    // Get index of max coefficient of output as the predicted class.
    Eigen::VectorXd::Index output_index;
    output.maxCoeff(&output_index);

    Eigen::VectorXd::Index label_index;
    label.maxCoeff(&label_index);

    if (output_index == label_index) {
      ++num_correct;
    }

    // std::cerr << "output: " << output.transpose() << std::endl;
    // std::cerr << "output index: " << output_index << std::endl;
    // std::cerr << "label: " << label.transpose() << std::endl;
    // std::cerr << "label index: " << label_index << std::endl;
    // std::cin.get();
  }
  std::cerr << "Correctly classified " << 100 * num_correct / num_eval << "%"
            << std::endl;
}

NeuralNetworkParameters Train(
    const NeuralNetworkParameters& nn_initial,
    const std::vector<Eigen::VectorXd>& training_inputs,
    const std::vector<Eigen::VectorXd>& training_labels,
    const std::vector<Eigen::VectorXd>& test_inputs,
    const std::vector<Eigen::VectorXd>& test_labels) {
  // TODO: Move these parameters elsewhere.
  double step_size = 5e-2;
  const int max_iterations = 10000;
  const int mini_batch_size = 100;
  const LossFunction loss_function = LossFunction::CROSS_ENTROPY;

  // Initialize Adam optimizer parameters.
  AdamOptimizerParameters adam;
  adam.weight_first_moment.resize(nn_initial.weights.size());
  adam.weight_second_moment.resize(nn_initial.weights.size());
  adam.bias_first_moment.resize(nn_initial.weights.size());
  adam.bias_second_moment.resize(nn_initial.weights.size());
  for (size_t i = 0; i < nn_initial.weights.size(); ++i) {
    adam.weight_first_moment.at(i) = Eigen::MatrixXd::Zero(
        nn_initial.weights.at(i).rows(), nn_initial.weights.at(i).cols());
    adam.weight_second_moment.at(i) = Eigen::MatrixXd::Zero(
        nn_initial.weights.at(i).rows(), nn_initial.weights.at(i).cols());
    adam.bias_first_moment.at(i) =
        Eigen::VectorXd::Zero(nn_initial.biases.at(i).size());
    adam.bias_second_moment.at(i) =
        Eigen::VectorXd::Zero(nn_initial.biases.at(i).size());
  }

  NeuralNetworkParameters nn = nn_initial;

  for (int iteration = 0; iteration < max_iterations; ++iteration) {
    // Select a random mini-batch of data points.
    const std::vector<size_t> random_indices =
        GenerateRandomIndices(training_inputs.size(), mini_batch_size);

    // Compute gradients for each element of mini-batch.
    // TODO: Reuse rather than reallocating on every loop.
    std::vector<Eigen::MatrixXd> weight_gradients_mini_batch(nn.weights.size());
    std::vector<Eigen::VectorXd> bias_gradients_mini_batch(nn.biases.size());
    for (size_t i = 0; i < nn.weights.size(); ++i) {
      const Eigen::MatrixXd& W = nn.weights.at(i);
      weight_gradients_mini_batch.at(i) =
          Eigen::MatrixXd::Zero(W.rows(), W.cols());
    }
    for (size_t i = 0; i < nn.biases.size(); ++i) {
      const Eigen::VectorXd& b = nn.biases.at(i);
      bias_gradients_mini_batch.at(i) = Eigen::VectorXd::Zero(b.size());
    }

    Eigen::VectorXd mini_batch_loss = Eigen::VectorXd::Zero(1);
    for (size_t i : random_indices) {
      Eigen::VectorXd loss;
      std::vector<Eigen::MatrixXd> weight_gradients;
      std::vector<Eigen::VectorXd> bias_gradients;
      EvaluateNetworkLossCombinedImplementation(
          training_inputs[i], nn, training_labels[i], loss_function, &loss,
          &weight_gradients, &bias_gradients);
      mini_batch_loss += loss;

      // Add gradient values for each element in the mini-batch.
      for (size_t j = 0; j < weight_gradients.size(); ++j) {
        weight_gradients_mini_batch.at(j) += weight_gradients.at(j);
      }
      for (size_t j = 0; j < bias_gradients.size(); ++j) {
        bias_gradients_mini_batch.at(j) += bias_gradients.at(j);
      }
    }
    mini_batch_loss /= mini_batch_size;

    // Add gradient values for each element in the mini-batch.
    for (size_t j = 0; j < weight_gradients_mini_batch.size(); ++j) {
      weight_gradients_mini_batch.at(j) /= mini_batch_size;
    }
    for (size_t j = 0; j < bias_gradients_mini_batch.size(); ++j) {
      bias_gradients_mini_batch.at(j) /= mini_batch_size;
    }

    std::cerr << "Iteration: " << iteration
              << ", Mini-Batch Loss: " << mini_batch_loss << std::endl;

    // Take a step in the gradient direction.
    // for (size_t j = 0; j < weight_gradients_mini_batch.size(); ++j) {
    //   nn.weights.at(j) +=
    //       -1 * step_size * weight_gradients_mini_batch.at(j);
    // }
    // for (size_t j = 0; j < bias_gradients_mini_batch.size(); ++j) {
    //   nn.biases.at(j) +=
    //       -1 * step_size * bias_gradients_mini_batch.at(j);
    // }

    AdamOptimizerParameters adam_update;
    NeuralNetworkParameters nn_update;
    AdamOptimizer(nn, adam, weight_gradients_mini_batch,
                  bias_gradients_mini_batch, iteration, &nn_update,
                  &adam_update);
    nn = nn_update;
    adam = adam_update;

    if (iteration > 0 && iteration % 100 == 0) {
      ComputePerformanceOnTestSet(nn, test_inputs, test_labels);
    }
  }
  return nn;
}

#include <algorithm>
#include <iostream>
#include <random>

#include "training.hpp"

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

NeuralNetworkParameters Train(
    const NeuralNetworkParameters& nn,
    const std::vector<Eigen::VectorXd>& training_inputs,
    const std::vector<Eigen::VectorXd>& training_labels) {
  // TODO: Move these parameters elsewhere.
  const double step_size = 5e-2;
  const int max_iterations = 1000;
  const int mini_batch_size = 50;
  const LossFunction loss_function = LossFunction::CROSS_ENTROPY;

  NeuralNetworkParameters nn_update = nn;
  for (int i = 0; i < max_iterations; ++i) {
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
      EvaluateNetworkLoss(training_inputs[i], nn_update, training_labels[i],
                          loss_function, &loss, &weight_gradients,
                          &bias_gradients);
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

    std::cerr << "Mini-Batch Loss: " << mini_batch_loss << std::endl;

    // Take a step in the gradient direction.
    for (size_t j = 0; j < weight_gradients_mini_batch.size(); ++j) {
      nn_update.weights.at(j) +=
          -1 * step_size * weight_gradients_mini_batch.at(j);
    }
    for (size_t j = 0; j < bias_gradients_mini_batch.size(); ++j) {
      nn_update.biases.at(j) +=
          -1 * step_size * bias_gradients_mini_batch.at(j);
    }
  }
}
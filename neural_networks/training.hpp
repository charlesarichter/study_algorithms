#pragma once

#include "nn.hpp"

std::vector<size_t> GenerateRandomIndices(const size_t max_index,
                                          const size_t num_indices);

void ComputePerformanceOnTestSet(
    const NeuralNetworkParameters& nn,
    const std::vector<Eigen::VectorXd>& test_inputs,
    const std::vector<Eigen::VectorXd>& test_labels);

NeuralNetworkParameters Train(
    const NeuralNetworkParameters& nn,
    const std::vector<Eigen::VectorXd>& training_inputs,
    const std::vector<Eigen::VectorXd>& training_labels,
    const std::vector<Eigen::VectorXd>& test_inputs,
    const std::vector<Eigen::VectorXd>& test_labels);

struct AdamOptimizerParameters {
  std::vector<Eigen::MatrixXd> weight_first_moment = {};
  std::vector<Eigen::MatrixXd> weight_second_moment = {};
  std::vector<Eigen::VectorXd> bias_first_moment = {};
  std::vector<Eigen::VectorXd> bias_second_moment = {};
};

void AdamOptimizer(const NeuralNetworkParameters& current_network_params,
                   const AdamOptimizerParameters& current_optimizer_params,
                   const std::vector<Eigen::MatrixXd>& weight_gradients,
                   const std::vector<Eigen::VectorXd>& bias_gradients,
                   const int iteration,
                   NeuralNetworkParameters* updated_network_params,
                   AdamOptimizerParameters* updated_optimizer_params);

#pragma once

#include "nn.hpp"

std::vector<size_t> GenerateRandomIndices(const size_t max_index,
                                          const size_t num_indices);

NeuralNetworkParameters Train(
    const NeuralNetworkParameters& nn,
    const std::vector<Eigen::VectorXd>& training_inputs,
    const std::vector<Eigen::VectorXd>& training_labels);

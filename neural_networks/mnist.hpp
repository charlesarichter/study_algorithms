#pragma once

#include <eigen3/Eigen/Dense>
#include <string>
#include <vector>

void LoadMnist(const std::string& csv_filename, const size_t num_images,
               std::vector<Eigen::VectorXd>* images,
               std::vector<Eigen::VectorXd>* labels);

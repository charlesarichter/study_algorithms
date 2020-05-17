#include "mnist.hpp"

#include <fstream>
#include <iostream>

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

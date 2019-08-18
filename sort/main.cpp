#include <iostream>
#include <random>
#include <vector>

#include "sort.hpp"

int main() {
  // Parameters to generate input vector of integers.
  const int min_val = -10;  // Lowest allowable value.
  const int max_val = 10;   // Highest allowable value.
  const int num_vals = 10;  // Number of values.

  std::random_device rd;
  std::uniform_int_distribution<int> dist(min_val, max_val);

  std::vector<int> input;
  for (int i = 0; i < num_vals; ++i) {
    // Generate random number.
    input.emplace_back(dist(rd));
  }

  std::cerr << "Input:";
  for (const int i : input) {
    std::cerr << " " << i;
  }
  std::cerr << std::endl;

  const std::vector<int> insertion_sort_result = InsertionSort(input);
  std::cerr << "Result:";
  for (const int i : insertion_sort_result) {
    std::cerr << " " << i;
  }
  std::cerr << std::endl;

  return 0;
}

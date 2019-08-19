#include <iostream>
#include <random>
#include <vector>

#include "sort.hpp"

void PrintVector(const std::string& name, const std::vector<int>& vector) {
  std::cerr << name;
  for (const int i : vector) {
    std::cerr << " " << i;
  }
  std::cerr << std::endl;
}

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
  PrintVector("Input", input);

  const std::vector<int> insertion_sort_result = InsertionSort(input);
  PrintVector("InsertionSort", insertion_sort_result);
  const std::vector<int> selection_sort_result = SelectionSort(input);
  PrintVector("SelectionSort", selection_sort_result);
  const std::vector<int> bubble_sort_result = BubbleSort(input);
  PrintVector("BubbleSort   ", bubble_sort_result);
  const std::vector<int> merge_sort_result = MergeSort(input);
  PrintVector("MergeSort    ", merge_sort_result);
  return 0;
}

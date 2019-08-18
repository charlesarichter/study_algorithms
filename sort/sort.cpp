#include "sort.hpp"

std::vector<int> InsertionSort(const std::vector<int>& input) {
  // Make a copy of the input so we can sort in place.
  std::vector<int> A = input;
  // From the second element (first is already sorted), iterate up the array.
  for (int i = 1; i < A.size(); ++i) {
    // Iterate back down the array towards the first element, looking for where
    // the value at A.at(i) belongs, swapping as we go.
    int j = i;
    while (j > 0 && (A.at(j - 1) > A.at(j))) {
      // Swap element at j-1 with element at j.
      const int tmp = A.at(j);
      A.at(j) = A.at(j - 1);
      A.at(j - 1) = tmp;
      // Step down the array and repeat.
      --j;
    }
  }
  return A;
}

void BubbleSort(const std::vector<int>& input) {}

void SelectionSort(const std::vector<int>& input) {}

void MergeSort(const std::vector<int>& input) {}

void QuickSort(const std::vector<int>& input) {}

void HeapSort(const std::vector<int>& input) {}

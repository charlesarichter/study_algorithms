#include "sort.hpp"

#include <limits>

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

std::vector<int> SelectionSort(const std::vector<int>& input) {
  // Make a copy of the input so we can sort in place.
  std::vector<int> A = input;

  // The variable i indicates the boundary between sorted and unsorted portions
  // A. Indices [i, A.size()-1] are unsorted, and [0, i-1] are sorted.
  for (int i = 0; i < A.size(); ++i) {
    // Find the smallest element in the unsorted portion [i, A.size()-1].
    int min_val = std::numeric_limits<int>::max();
    int min_ind = 0;
    for (int j = i; j < A.size(); ++j) {
      if (A.at(j) < min_val) {
        min_ind = j;
        min_val = A.at(j);
      }
    }

    // Now we should have the index of the smallest unsorted element.
    if (min_ind != i) {
      int tmp = A.at(i);
      A.at(i) = A.at(min_ind);
      A.at(min_ind) = tmp;
    }
    // Increment the boundary between sorted and unsorted.
  }
  return A;
}

void BubbleSort(const std::vector<int>& input) {}

void MergeSort(const std::vector<int>& input) {}

void QuickSort(const std::vector<int>& input) {}

void HeapSort(const std::vector<int>& input) {}

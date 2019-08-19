#include "sort.hpp"

#include <iostream>
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

std::vector<int> HeapSort(const std::vector<int>& input) {}

std::vector<int> BubbleSort(const std::vector<int>& input) {
  // Make a copy of the input so we can sort in place.
  std::vector<int> A = input;

  // Loop over the whole array until we loop and nothing changes.
  while (true) {
    bool changed = false;
    for (int i = 1; i < A.size(); ++i) {
      // If adjacent elements are out of order, swap them.
      if (A.at(i) < A.at(i - 1)) {
        const int tmp = A.at(i);
        A.at(i) = A.at(i - 1);
        A.at(i - 1) = tmp;
        changed = true;
      }
    }
    if (!changed) {
      break;
    }
  }
  return A;
}

// TODO: Also implement this with the input arguments in CLRS (Intro to Algs)
static std::vector<int> Merge(const std::vector<int>& L,
                              const std::vector<int>& R) {
  const int n = L.size() + R.size();

  // Copy input and add "sentinel" value.
  std::vector<int> l_sentinel = L;
  std::vector<int> r_sentinel = R;
  l_sentinel.emplace_back(std::numeric_limits<int>::max());
  r_sentinel.emplace_back(std::numeric_limits<int>::max());

  int l_counter = 0;
  int r_counter = 0;
  int values_processed = 0;
  std::vector<int> result;  //(n);
  while (values_processed < n) {
    if (l_sentinel.at(l_counter) < r_sentinel.at(r_counter)) {
      result.emplace_back(l_sentinel.at(l_counter));
      ++l_counter;
    } else {
      result.emplace_back(r_sentinel.at(r_counter));
      ++r_counter;
    }
    ++values_processed;
  }
  return result;
}

std::vector<int> MergeSort(const std::vector<int>& input) {
  // Make a copy of the input to operate on.
  std::vector<int> A = input;

  // If the input has size 1, then it's already sorted and we can return it.
  if (input.size() == 1) {
    return A;
  }

  // std::cerr << "MergeSort input size: " << input.size() << std::endl;

  // Divide the input in half.
  const int n = input.size();
  const std::vector<int> L(input.begin(), input.begin() + n / 2);
  const std::vector<int> R(input.begin() + n / 2, input.end());

  // std::cerr << "n: " << n << " n/2: " << n / 2 << std::endl;
  // std::cerr << "L";
  // for (const int i : L) {
  //   std::cerr << " " << i;
  // }
  // std::cerr << std::endl;
  // std::cerr << "R";
  // for (const int i : R) {
  //   std::cerr << " " << i;
  // }
  // std::cerr << std::endl;

  // (Recursively) Sort the left and right halves.
  const std::vector<int> l_sorted = MergeSort(L);
  const std::vector<int> r_sorted = MergeSort(R);

  // std::cerr << "L Sorted";
  // for (const int i : l_sorted) {
  //   std::cerr << " " << i;
  // }
  // std::cerr << std::endl;
  // std::cerr << "R Sorted";
  // for (const int i : r_sorted) {
  //   std::cerr << " " << i;
  // }
  // std::cerr << std::endl;

  // Combine the sorted halves with the Merge function.
  return Merge(l_sorted, r_sorted);
}

void QuickSort(const std::vector<int>& input) {}

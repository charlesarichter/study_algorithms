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
static std::vector<int> Merge(const std::vector<int>& l_input,
                              const std::vector<int>& r_input) {
  const int n = l_input.size() + r_input.size();

  // Copy input and add "sentinel" value.
  std::vector<int> l = l_input;
  std::vector<int> r = r_input;
  l.emplace_back(std::numeric_limits<int>::max());
  r.emplace_back(std::numeric_limits<int>::max());

  int l_counter = 0;
  int r_counter = 0;
  int values_processed = 0;
  std::vector<int> result;
  while (values_processed < n) {
    if (l.at(l_counter) < r.at(r_counter)) {
      result.emplace_back(l.at(l_counter));
      ++l_counter;
    } else {
      result.emplace_back(r.at(r_counter));
      ++r_counter;
    }
    ++values_processed;
  }
  return result;
}

std::vector<int> MergeSort(const std::vector<int>& input) {
  // If the input has size 1, then it's already sorted and we can return it.
  if (input.size() == 1) {
    return input;
  }

  // Divide the input in half.
  const int n = input.size();
  const std::vector<int> l(input.begin(), input.begin() + n / 2);
  const std::vector<int> r(input.begin() + n / 2, input.end());

  // (Recursively) Sort the left and right halves.
  const std::vector<int> l_sorted = MergeSort(l);
  const std::vector<int> r_sorted = MergeSort(r);

  // Combine the sorted halves with the Merge function.
  return Merge(l_sorted, r_sorted);
}

static int QuickSortPartitionHoare(std::vector<int>& input, const int lo_input,
                                   const int hi_input) {
  // These indices start off at the boundaries of the input, and will work their
  // way towards each other (swapping input values) until they reach the pivot.
  int lo = lo_input;
  int hi = hi_input;

  // The pivot value is the value at the mid point of the input.
  const int pivot = input.at(lo + (hi - lo) / 2);

  while (true) {
    // Increment lo until we reach an element that input(lo) is not less than.
    while (input.at(lo) < pivot) {
      ++lo;
    }

    // Decrement hi until we get to an element that input(hi) is not more than.
    while (input.at(hi) > pivot) {
      --hi;
    }

    // If lo and hi have crossed, or are equal, we're done. That means we have
    // swapped all elements that need to be swapped in order for all elements
    // left of the pivot to be less than or equal to the pivot, and all elements
    // right of the pivot to be greater than or equal to the pivot.
    if (lo >= hi) {
      return hi;
    }

    // When we reach this point, lo and hi are indices of elements that are on
    // the "wrong side" of the pivot. Swap.
    const int tmp = input.at(hi);
    input.at(hi) = input.at(lo);
    input.at(lo) = tmp;
    ++lo;
    --hi;
  }
}

void QuickSort(std::vector<int>& input, const int lo, const int hi) {
  if (lo < hi) {
    const int p = QuickSortPartitionHoare(input, lo, hi);
    QuickSort(input, lo, p);
    QuickSort(input, p + 1, hi);
  }
}

#pragma once

#include <vector>

// One of the most basic sorting algorithms, sorts in place.
//
// The outer loop iterates from "left to right" along the array.
//
// The inner loop takes the current element and pushes it "right to left" back
// down the array until it reaches an element that is smaller than it. While the
// element is pushed right to left, it swaps with each element it encounters
// that is out of order.
//
// After k passes through the array, the first k (or rather k + 1) elements are
// sorted (although, those are not necessarily the smallest k elements.
//
// Time Complexity is O(n^2): For each element (outer loop) it checks against
// each other element (inner loop). Space Complexity is n: Sorts in-place,
// with a temporary single storage place for one element while swapping.
std::vector<int> InsertionSort(const std::vector<int>& input);

// Simple, in-place, O(n^2) time complexity.
//
// Maintain a boundary between sorted (left side) and unsorted (right side)
// portion of input array. In each iteration, find the smallest element in the
// unsorted portion and swap it with the left-most element in the unsorted
// portion of the array. Then move the boundary one place to the right.
std::vector<int> SelectionSort(const std::vector<int>& input);

// The same as SelectionSort, only uses a heap to find the smallest element in
// the unsorted portion of the array.
std::vector<int> HeapSort(const std::vector<int>& input);

// Very simple...not really a serious algorithm, but well known because of its
// simplicity. In-place, O(n^2) time complexity.
//
// Iterate through the input array repeatedly, swapping adjacent elements if
// they are out of order, until no more swaps are needed. A maximum of n passes
// through the array may be required.
std::vector<int> BubbleSort(const std::vector<int>& input);

std::vector<int> MergeSort(const std::vector<int>& input);

void QuickSort(const std::vector<int>& input);

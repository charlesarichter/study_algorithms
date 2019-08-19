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

// A Divide-and-Conquer approch. O(n log n) time complexity.
//
// TODO: Derive time and space complexity.
//
// MergeSort takes its input, divides it into two sub-arrays, sorts them, and
// re-combines them into a single sorted array. And the method it uses to sort
// each sub-array is...itself. It's a recursive algorithm. The recursion bottoms
// out when the sub-arrays are of size 1 and are therefore trivially sorted.
//
// The method it uses to combine the two sorted halves into a full sorted array
// is an auxiliary function called Merge, which simply combines the two inputs,
// one element at a time, knowing that they are sorted. Imagine combining two
// sorted piles of cards into a single sorted pile of cards. You just look at
// which card is at the top of each pile, take the smaller one and put it into
// the output pile and repeat until you're done.
std::vector<int> MergeSort(const std::vector<int>& input);

void QuickSort(const std::vector<int>& input);

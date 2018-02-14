#include <iostream>
#include <vector>
#include <random>

class BinaryHeap {
 public:
  BinaryHeap();
  ~BinaryHeap();

  void Push(const double in);
  double PopMin();
  void BubbleUp(const int ind);
  void BubbleDown(const int ind);

  int GetParentIndex(const int ind);
  int GetLeftChildIndex(const int ind);
  int GetRightChildIndex(const int ind);

 private:

  std::vector<double> data_;
};

BinaryHeap::BinaryHeap() {

  // Add a filler element to data_ so that the simple count-from-one indexing
  // arithmetic works.
  data_.push_back(0);
}

BinaryHeap::~BinaryHeap() {
}

int BinaryHeap::GetParentIndex(const int ind) {
  if (ind % 2 == 0) {
    return ind / 2;
  } else {
    return (ind - 1) / 2;
  }
}

int BinaryHeap::GetLeftChildIndex(const int ind) {
  return 2 * ind;
}

int BinaryHeap::GetRightChildIndex(const int ind) {
  return 2 * ind + 1;
}

void BinaryHeap::Push(const double in) {

  // std::cerr << "Adding " << in << std::endl;

  // Insert the new element at the end
  data_.push_back(in);
  int i_added = data_.size() -1;

  // std::cerr << "Index just added: " << i_added << std::endl;

  // Then, bubble it up: Compare with parent, and swap if necessary
  BubbleUp(i_added);
}

void BinaryHeap::BubbleUp(const int ind) {

  // If this is the top of the heap, do nothing
  if (ind == 1) {
    // std::cerr << "Ind = 1" << std::endl;
    return;
  }

  // Otherwise, find the parent of ind and swap if necessary
  int parent_ind = GetParentIndex(ind);
  // std::cerr << "Parent of " << ind << " is " << parent_ind << std::endl;

  int parent_value = data_.at(parent_ind);
  int child_value = data_.at(ind);
  if (child_value < parent_value)  {
    // Swap
    data_.at(ind) = parent_value;
    data_.at(parent_ind) = child_value;

    // Now recursively try again with the next parent up
    BubbleUp(parent_ind);
  }
}

void BinaryHeap::BubbleDown(const int ind) {

  // Take the element at ind and swap it with either of its two children such
  // that the resulting parent is less than both of its children

  // Swap with the smaller of the two children

  int ind_left = GetLeftChildIndex(ind);
  int ind_right = GetRightChildIndex(ind);

  int ind_last = data_.size() - 1;
  
  // std::cerr << "Bubble down index " << ind << std::endl;
  // std::cerr << "Left child index " << ind_left << std::endl;
  // std::cerr << "Right child index " << ind_right << std::endl;

  if (ind_left > ind_last) {
    // Even the lower of the two child indices is beyond the actual data
    // structure, so we have hit the bottom of the tree.
    return;
  } 

  double value_parent = data_.at(ind);
  double value_left = data_.at(ind_left);

  if (ind_left == ind_last) {
    // We only have one child in this case
    if (value_left < value_parent) {
      data_.at(ind_left) = value_parent;
      data_.at(ind) = value_left;

      // TODO/FIXME: I think if we have only one child, we should be in the
      // bottom rung of the tree, so further calls to BubbleDown should do
      // nothing...but would be best to confirm this.
      BubbleDown(ind_left);
    }
    return;
  }

  // If we have made it this far, there are two children
  double value_right = data_.at(ind_right);
  double value_diff_left = value_left - value_parent;
  double value_diff_right = value_right - value_parent;

  // std::cerr << "Value parent " << value_parent << std::endl;
  // std::cerr << "Value left " << value_left << std::endl;
  // std::cerr << "Value right " << value_right << std::endl;
  // std::cerr << "Value diff left " << value_diff_left << std::endl;
  // std::cerr << "Value diff right " << value_diff_right << std::endl;

  if (value_diff_left < value_diff_right && value_diff_left < 0) {
    // Swap left and parent
    data_.at(ind_left) = value_parent;
    data_.at(ind) = value_left;
    BubbleDown(ind_left);
  } else if (value_diff_left > value_diff_right && value_diff_right < 0) {
    // Swap right and parent
    data_.at(ind_right) = value_parent;
    data_.at(ind) = value_right;
    BubbleDown(ind_right);
  } else {
    
    // assert(value_diff_left > 0 && value_diff_right > 0);
    // Parent is less than or equal to both children. No swap needed.
  }
}

double BinaryHeap::PopMin() {

  // Remove the top of the heap, move the last element into the top position,
  // then bubble it down
  if (data_.size() == 1) {
    std::cerr << "BinaryHeap is empty!" << std::endl;
    return 0;
  }

  // Get the min value
  double min = data_.at(1);

  // Move the last element of the array into the top position
  data_.at(1) = data_.back();
  data_.pop_back();

  // Bubble down the top position
  BubbleDown(1);

  return min;
}

int main() {

  BinaryHeap bh;

  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(0.0,100.0);

  int nrolls = 10;
  for (int i=0; i<nrolls; ++i) {
    double number = distribution(generator);
    std::cerr << "Number: " << number << std::endl;
    bh.Push(number);
  }

  for (int i=0; i<nrolls; ++i) {
    std::cerr << "Min: " << bh.PopMin() << std::endl;
  }

  // bh.Push(4);
  // bh.Push(1);
  // bh.Push(9);
  // bh.Push(3);
  // bh.Push(0);
  // bh.Push(-20);
  // bh.Push(16);
  // bh.Push(5);
  //
  // std::cerr << "Min: " << bh.PopMin() << std::endl;
  // std::cerr << "Min: " << bh.PopMin() << std::endl;
  // std::cerr << "Min: " << bh.PopMin() << std::endl;
  // std::cerr << "Min: " << bh.PopMin() << std::endl;
  // std::cerr << "Min: " << bh.PopMin() << std::endl;
  // std::cerr << "Min: " << bh.PopMin() << std::endl;
  // std::cerr << "Min: " << bh.PopMin() << std::endl;
  // std::cerr << "Min: " << bh.PopMin() << std::endl;

  return 0;
}

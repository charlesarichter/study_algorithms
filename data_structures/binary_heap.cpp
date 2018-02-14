#include <iostream>
#include <vector>

class BinaryHeap {
 public:
  BinaryHeap();
  ~BinaryHeap();

  void Push(const double in);
  double PopMin();

 private:

  std::vector<double> data_;
};

BinaryHeap::BinaryHeap() {
  data_.push_back(0);
}

BinaryHeap::~BinaryHeap() {
}

void BinaryHeap::Push(const double in) {

  std::cerr << "Adding " << in << std::endl;

  // Insert the new element at the end
  // data_.push_back(in);

  // Then, bubble it up
}

double BinaryHeap::PopMin() {

  // Remove the top of the heap, move the last element into the top position,
  // then bubble it down

  return 0;
}

int main() {

  BinaryHeap bh;

  bh.Push(1);
  bh.Push(4);
  bh.Push(9);
  bh.Push(3);

  double min = bh.PopMin();
  // std::cerr << "Min: " << min << std::endl;

  return 0;
}

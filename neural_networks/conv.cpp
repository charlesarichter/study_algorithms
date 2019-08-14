#include "conv.hpp"

void Conv(const std::vector<Eigen::MatrixXd>& input_volume,
          const std::vector<std::vector<Eigen::MatrixXd>>& conv_kernels,
          std::vector<Eigen::MatrixXd>* output_volume) {}

void TestConv() {
  // Test from example in: http://cs231n.github.io/convolutional-networks/

  // clang-format off

  // Input Volume (+pad 1) (7x7x3)
  // x[:,:,0]
  Eigen::MatrixXd x0(7, 7);
  x0 << 0, 0, 0, 0, 0, 0, 0, 
        0, 2, 1, 2, 2, 2, 0, 
        0, 2, 2, 1, 0, 0, 0, 
        0, 2, 1, 1, 2, 2, 0, 
        0, 2, 1, 1, 2, 0, 0, 
        0, 0, 1, 2, 0, 2, 0, 
        0, 0, 0, 0, 0, 0, 0;
  // x[:,:,1]
  Eigen::MatrixXd x1(7, 7);
  x1 << 0, 0, 0, 0, 0, 0, 0, 
        0, 1, 2, 0, 2, 1, 0, 
        0, 1, 1, 0, 0, 0, 0, 
        0, 0, 1, 1, 0, 0, 0, 
        0, 2, 0, 1, 0, 2, 0, 
        0, 2, 0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 0, 0;
  // x[:,:,2]
  Eigen::MatrixXd x2(7, 7);
  x2 << 0, 0, 0, 0, 0, 0, 0, 
        0, 0, 2, 0, 2, 2, 0, 
        0, 2, 1, 2, 2, 2, 0, 
        0, 0, 1, 1, 2, 2, 0, 
        0, 1, 2, 0, 1, 2, 0, 
        0, 2, 1, 2, 0, 0, 0, 
        0, 0, 0, 0, 0, 0, 0;

  // Filter W0 (3x3x3)
  // w0[:,:,0]
  Eigen::MatrixXd w00(3,3);
  w00 <<  0, -1, -1,
         -1,  0, -1, 
         -1,  0,  0;

  // w0[:,:,1],
  Eigen::MatrixXd w01(3,3);
  w01 << -1,  0, -1, 
          0,  1, -1, 
         -1, -1,  0;

  // w0[:,:,2],
  Eigen::MatrixXd w02(3,3);
  w02 <<  0,  1, -1, 
          0,  0,  1, 
         -1,  0,  1;

   // Bias b0 (1x1x1),
   // b0[:,:,0],
  Eigen::MatrixXd b0(1,1);
  b0 << 1;

  // Filter W1 (3x3x3),
  // w1[:,:,0],
  Eigen::MatrixXd w10(3,3);
  w10 <<  1,  0, -1, 
          1, -1, -1, 
          0,  1,  0;

  // w1[:,:,1],
  Eigen::MatrixXd w11(3,3);
  w11 <<  0,  0,  1, 
          1, -1,  0, 
         -1, -1,  0;

  // w1[:,:,2],
  Eigen::MatrixXd w12(3,3);
  w12 <<  0,  1, -1, 
         -1,  1,  0, 
         -1,  1,  1;

  // Bias b1 (1x1x1),
  // b1[:,:,0],
  Eigen::MatrixXd b1(1,1);
  b1 << 0;

  // Output Volume (3x3x2),
  // o[:,:,0],
  Eigen::MatrixXd o0(3,3);
  o0 <<  1, -4, -2, 
        -4, -4, -4, 
        -1, -4,  3;

  // o[:,:,1],
  Eigen::MatrixXd o1(3,3);
  o1 <<  0,  0,  1, 
         0, -1,  1, 
        -3, -2,  2;

  // clang-format on
}

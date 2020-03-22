#include <iostream>

#include "conv_example.hpp"

ConvExample GetConvExample1() {
  // Test from example in: http://cs231n.github.io/convolutional-networks/

  // clang-format off

  // Input Volume (+pad 1) (7x7x3)
  // x[:,:,0]
  // Eigen::MatrixXd x0(7, 7);
  // x0 << 0, 0, 0, 0, 0, 0, 0,
	// 			0, 2, 2, 2, 2, 0, 0,
	// 			0, 1, 2, 1, 1, 1, 0,
	// 			0, 2, 1, 1, 1, 2, 0,
	// 			0, 2, 0, 2, 2, 0, 0,
	// 			0, 2, 0, 2, 0, 2, 0,
	// 			0, 0, 0, 0, 0, 0, 0;
  //
  // // x[:,:,1]
  // Eigen::MatrixXd x1(7, 7);
  // x1 << 0, 0, 0, 0, 0, 0, 0,
  //       0, 1, 1, 0, 2, 2, 0,
  //       0, 2, 1, 1, 0, 0, 0,
  //       0, 0, 0, 1, 1, 0, 0,
  //       0, 2, 0, 0, 0, 0, 0,
  //       0, 1, 0, 0, 2, 0, 0,
  //       0, 0, 0, 0, 0, 0, 0;
  //
  // // x[:,:,2]
  // Eigen::MatrixXd x2(7, 7);
  // x2 << 0, 0, 0, 0, 0, 0, 0,
  //       0, 0, 2, 0, 1, 2, 0,
  //       0, 2, 1, 1, 2, 1, 0,
  //       0, 0, 2, 1, 0, 2, 0,
  //       0, 2, 2, 2, 1, 0, 0,
  //       0, 2, 2, 2, 2, 0, 0,
  //       0, 0, 0, 0, 0, 0, 0;

  // Input volume without padding (to be added in the Conv function).
  // x[:,:,0]
  Eigen::MatrixXd x0(5, 5);
	x0 << 2, 2, 2, 2, 0,
				1, 2, 1, 1, 1,
				2, 1, 1, 1, 2,
				2, 0, 2, 2, 0,
				2, 0, 2, 0, 2;

  // x[:,:,1]
  Eigen::MatrixXd x1(5, 5);
  x1 << 1, 1, 0, 2, 2,
        2, 1, 1, 0, 0,
        0, 0, 1, 1, 0,
        2, 0, 0, 0, 0,
        1, 0, 0, 2, 0;

  // x[:,:,2]
  Eigen::MatrixXd x2(5, 5);
  x2 << 0, 2, 0, 1, 2,
        2, 1, 1, 2, 1,
        0, 2, 1, 0, 2,
        2, 2, 2, 1, 0,
        2, 2, 2, 2, 0;

  // Filter W0 (3x3x3)
  // w0[:,:,0]
  Eigen::MatrixXd w00(3,3);
  w00 <<  0, -1, -1,
         -1,  0,  0,
         -1, -1,  0;

  // w0[:,:,1],
  Eigen::MatrixXd w01(3,3);
  w01 << -1,  0, -1,
          0,  1, -1,
         -1, -1,  0;

  // w0[:,:,2],
  Eigen::MatrixXd w02(3,3);
  w02 <<  0,  0, -1,
          1,  0,  0,
         -1,  1,  1;

  // Bias b0 (1x1x1),
  // b0[:,:,0],
  const double b0 = 1;

  // Filter W1 (3x3x3),
  // w1[:,:,0],
  Eigen::MatrixXd w10(3,3);
  w10 <<  1,  1,  0,
          0, -1,  1,
         -1, -1,  0;

  // w1[:,:,1],
  Eigen::MatrixXd w11(3,3);
  w11 <<  0,  1, -1,
          0, -1, -1,
          1,  0,  0;

  // w1[:,:,2],
  Eigen::MatrixXd w12(3,3);
  w12 <<  0, -1, -1,
          1,  1,  1,
         -1,  0,  1;

  // Bias b1 (1x1x1),
  // b1[:,:,0],
  const double b1 = 0;

  // Output Volume (3x3x2),
  // o[:,:,0],
  Eigen::MatrixXd o0(3,3);
  o0 <<  1, -4, -1,
        -4, -4, -4,
        -2, -4,  3;

  // o[:,:,1],
  Eigen::MatrixXd o1(3,3);
  o1 <<  0,  0, -3,
         0, -1, -2,
         1,  1,  2;

  // clang-format on

  const std::vector<Eigen::MatrixXd> output_volume_expected{o0, o1};

  // Input volume is 3-channel.
  const std::vector<Eigen::MatrixXd> input_volume{x0, x1, x2};

  // Each of the N (in ths case 2) conv filters must have as many channels as
  // the input (in this case 3).
  //
  // The depth of each conv filter must equal the depth of the input volume and
  // the number of conv filters determines the depth of the output volume.
  const std::vector<std::vector<Eigen::MatrixXd>> conv_kernels{{w00, w01, w02},
                                                               {w10, w11, w12}};

  // TODO: Should biases be matrices? Will they always be 1x1?
  const std::vector<double> biases{b0, b1};

  // Padding of 1 for this example.
  const int padding = 1;

  // Stride of 2 for this example.
  const int stride = 2;

  return ConvExample{input_volume, conv_kernels, biases,
                     stride,       padding,      output_volume_expected};
}

ConvExample GetConvExample2() {
  // Test from example in: http://cs231n.github.io/convolutional-networks/

  // clang-format off

  // Input Volume (+pad 1) (7x7x3)
  // x[:,:,0]
  // Eigen::MatrixXd x0(7, 7);
  // x0 << 0, 0, 0, 0, 0, 0, 0,
	// 			0, 1, 2, 2, 0, 1, 0,
	// 			0, 0, 0, 2, 2, 0, 0,
	// 			0, 0, 1, 1, 2, 2, 0,
	// 			0, 1, 1, 2, 1, 1, 0,
	// 			0, 0, 0, 2, 2, 1, 0,
	// 			0, 0, 0, 0, 0, 0, 0;
  //
  // // x[:,:,1]
  // Eigen::MatrixXd x1(7, 7);
  // x1 << 0, 0, 0, 0, 0, 0, 0,
  //       0, 2, 0, 2, 0, 1, 0,
  //       0, 2, 2, 2, 2, 2, 0,
  //       0, 1, 2, 2, 1, 1, 0,
  //       0, 1, 0, 1, 1, 1, 0,
  //       0, 2, 0, 1, 2, 0, 0,
  //       0, 0, 0, 0, 0, 0, 0;
  //
  // // x[:,:,2]
  // Eigen::MatrixXd x2(7, 7);
  // x2 << 0, 0, 0, 0, 0, 0, 0,
  //       0, 2, 2, 0, 2, 2, 0,
  //       0, 0, 1, 2, 1, 1, 0,
  //       0, 2, 2, 1, 1, 0, 0,
  //       0, 0, 1, 0, 2, 2, 0,
  //       0, 2, 1, 2, 0, 1, 0,
  //       0, 0, 0, 0, 0, 0, 0;

  // Input volume without padding (to be added in the Conv function).
  // x[:,:,0]
  Eigen::MatrixXd x0(5, 5);
  x0 << 1, 2, 2, 0, 1,
				0, 0, 2, 2, 0,
				0, 1, 1, 2, 2,
				1, 1, 2, 1, 1,
				0, 0, 2, 2, 1;

  // x[:,:,1]
  Eigen::MatrixXd x1(5, 5);
  x1 << 2, 0, 2, 0, 1,
        2, 2, 2, 2, 2,
        1, 2, 2, 1, 1,
        1, 0, 1, 1, 1,
        2, 0, 1, 2, 0;

  // x[:,:,2]
  Eigen::MatrixXd x2(5, 5);
  x2 << 2, 2, 0, 2, 2,
        0, 1, 2, 1, 1,
        2, 2, 1, 1, 0,
        0, 1, 0, 2, 2,
        2, 1, 2, 0, 1;

  // Filter W0 (3x3x3)
  // w0[:,:,0]
  Eigen::MatrixXd w00(3,3);
  w00 << -1,  1, -1,
          0,  0, -1,
         -1,  1, -1;

  // w0[:,:,1],
  Eigen::MatrixXd w01(3,3);
  w01 <<  0,  0,  1,
         -1,  1,  1,
          1,  0,  0;

  // w0[:,:,2],
  Eigen::MatrixXd w02(3,3);
  w02 << -1,  0, -1,
          0,  0,  0,
         -1, -1, -1;

  // Bias b0 (1x1x1),
  // b0[:,:,0],
  const double b0 = 1;

  // Filter W1 (3x3x3),
  // w1[:,:,0],
  Eigen::MatrixXd w10(3,3);
  w10 <<  1,  1,  0,
          0,  0, -1,
          0, -1, -1;

  // w1[:,:,1],
  Eigen::MatrixXd w11(3,3);
  w11 << -1, -1, -1,
          1,  0,  1,
         -1, -1,  1;

  // w1[:,:,2],
  Eigen::MatrixXd w12(3,3);
  w12 << -1,  0,  0,
          0,  0,  0,
          0,  1,  1;

  // Bias b1 (1x1x1),
  // b1[:,:,0],
  const double b1 = 0;

  // Output Volume (3x3x2),
  // o[:,:,0],
  Eigen::MatrixXd o0(3,3);
  o0 <<  0,  1,  0,
         3, -3, -5,
         2,  0, -3;

  // o[:,:,1],
  Eigen::MatrixXd o1(3,3);
  o1 << -1, -3, -3,
        -5, -5, -3,
         0,  0,  0;

  // clang-format on

  const std::vector<Eigen::MatrixXd> output_volume_expected{o0, o1};

  // Input volume is 3-channel.
  const std::vector<Eigen::MatrixXd> input_volume{x0, x1, x2};

  // Each of the N (in ths case 2) conv filters must have as many channels as
  // the input (in this case 3).
  //
  // The depth of each conv filter must equal the depth of the input volume and
  // the number of conv filters determines the depth of the output volume.
  const std::vector<std::vector<Eigen::MatrixXd>> conv_kernels{{w00, w01, w02},
                                                               {w10, w11, w12}};

  // TODO: Should biases be matrices? Will they always be 1x1?
  const std::vector<double> biases{b0, b1};

  // Padding of 1 for this example.
  const int padding = 1;

  // Stride of 2 for this example.
  const int stride = 2;

  return ConvExample{input_volume, conv_kernels, biases,
                     stride,       padding,      output_volume_expected};
}

ConvExample GetConvExample3() {
  // Test from example in: https://knet.readthedocs.io/en/latest/cnn.html

  // clang-format off

  // Input Volume (+pad 0) (3x3x1)
  // x[:,:,0]
  Eigen::MatrixXd x0(3, 3);
  x0 << 1, 4, 7,
				2, 5, 8,
				3, 6, 9;

  // Filter W0 (2x2x1)
  // w0[:,:,0]
  Eigen::MatrixXd w00(2,2);
  w00 <<  4,  2,
          3,  1;

  // NOTE: In https://knet.readthedocs.io/en/latest/cnn.html, the kernel is
  // explicitly flipped from this form (convolution vs. cross-correlation).
  // w00 <<  1,  3,
  //         2,  4;

  // Bias b0 (1x1x1),
  // b0[:,:,0],
  const double b0 = 0;

  // Output Volume (2x2x1),
  // o[:,:,0],
  Eigen::MatrixXd o0(2,2);
  o0 <<  23, 53,
         33, 63;

  // clang-format on

  const std::vector<Eigen::MatrixXd> output_volume_expected{o0};

  // Input volume is 3-channel.
  std::vector<Eigen::MatrixXd> input_volume{x0};

  // Each of the N (in ths case 2) conv filters must have as many channels as
  // the input (in this case 3).
  //
  // The depth of each conv filter must equal the depth of the input volume and
  // the number of conv filters determines the depth of the output volume.
  std::vector<std::vector<Eigen::MatrixXd>> conv_kernels{{w00}};

  // TODO: Should biases be matrices? Will they always be 1x1?
  std::vector<double> biases{b0};

  // Padding of 0 for this example.
  const int padding = 0;

  // Stride of 1 for this example.
  const int stride = 1;

  return ConvExample{input_volume, conv_kernels, biases,
                     stride,       padding,      output_volume_expected};
}

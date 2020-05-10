#include "network_test.hpp"

#include <iostream>

#include "network.hpp"
#include "nn.hpp"

void RunNetworkTest() {
  // Define network.
  // const int num_conv_layers = 3;
  const int num_kernels = 3;
  const int kernel_size = 2;  // TODO: Enable non-square kernels.
  const int stride = 1;

  const int num_nodes = 20;

  const ActivationFunction activation_function = ActivationFunction::RELU;
  const ActivationFunction output_function = ActivationFunction::SOFTMAX;
  const LossFunction loss_function = LossFunction::CROSS_ENTROPY;

  // Define input.
  const int input_size = 10;  // TODO: Enable non-square inputs.
  const int num_input_channels = 3;

  // Define output.
  // Assume multi-class classification, one-hot representation.
  const int num_categories = 10;

  // Define layers.
  LayerConvPtr layer_0 = std::make_shared<LayerConv>(
      input_size, input_size, num_input_channels, kernel_size, kernel_size,
      num_kernels, stride, activation_function);

  const int layer_0_output_rows = layer_0->GetOutputRows();
  const int layer_0_output_cols = layer_0->GetOutputCols();
  const int layer_0_output_channels = layer_0->GetOutputChannels();

  LayerConvPtr layer_1 = std::make_shared<LayerConv>(
      layer_0_output_rows, layer_0_output_cols, layer_0_output_channels,
      kernel_size, kernel_size, num_kernels, stride, activation_function);

  const int layer_1_output_rows = layer_1->GetOutputRows();
  const int layer_1_output_cols = layer_1->GetOutputCols();
  const int layer_1_output_channels = layer_1->GetOutputChannels();

  LayerConvPtr layer_2 = std::make_shared<LayerConv>(
      layer_1_output_rows, layer_1_output_cols, layer_1_output_channels,
      kernel_size, kernel_size, num_kernels, stride, activation_function);

  const int layer_2_output_rows = layer_2->GetOutputRows();
  const int layer_2_output_cols = layer_2->GetOutputCols();
  const int layer_2_output_channels = layer_2->GetOutputChannels();

  const int layer_2_num_outputs =
      layer_2_output_rows * layer_2_output_cols * layer_2_output_channels;

  LayerFCPtr layer_3 = std::make_shared<LayerFC>(layer_2_num_outputs, num_nodes,
                                                 activation_function);
  LayerFCPtr layer_4 =
      std::make_shared<LayerFC>(num_nodes, num_nodes, activation_function);

  LayerFCPtr layer_5 =
      std::make_shared<LayerFC>(num_nodes, num_nodes, output_function);

  // Build network.
  Network network({layer_0, layer_1, layer_2, layer_3, layer_4, layer_5});

  // Get initial set of parameters.
  std::vector<double> parameters = network.GetRandomParameters();

  // Generate random input.
  const std::vector<double> input =
      GetRandomVector(input_size * input_size * num_input_channels, 0, 1);
  const std::vector<double> label;

  // Evaluate network.
  std::vector<double> gradient;
  const double loss = network.Evaluate(input, label, parameters, &gradient);
}

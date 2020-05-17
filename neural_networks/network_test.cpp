#include "network_test.hpp"

#include <iostream>

#include "network.hpp"
#include "nn.hpp"
#include "training.hpp"

static Network BuildTestNetwork(const int input_channels, const int input_rows,
                                const int input_cols, const int num_outputs) {
  // Specify conv layers.
  const int num_kernels = 3;
  const int kernel_size = 2;  // TODO: Enable non-square kernels.
  const int stride = 1;

  // Fully connected layer width.
  const int num_nodes = 20;

  const ActivationFunction activation_function = ActivationFunction::SIGMOID;
  const ActivationFunction output_function = ActivationFunction::SOFTMAX;

  // Define layers.
  LayerConvPtr layer_0 = std::make_shared<LayerConv>(
      input_rows, input_cols, input_channels, kernel_size, kernel_size,
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
      std::make_shared<LayerFC>(num_nodes, num_outputs, output_function);

  // Build network.
  return Network({layer_0, layer_1, layer_2, layer_3, layer_4, layer_5});
}

void RunNetworkGradientTest() {
  // Create network.
  const int input_size = 10;  // TODO: Enable non-square inputs.
  const int input_channels = 3;
  const int num_categories = 10;
  const Network network =
      BuildTestNetwork(input_channels, input_size, input_size, num_categories);

  // Get initial set of parameters.
  std::vector<double> parameters = network.GetRandomParameters();

  // Generate random input.
  // TODO: Consider scaling, centering, normalization of input.
  const std::vector<double> input =
      GetRandomVector(input_size * input_size * input_channels, 0, 1);

  // Generate arbitrary label.
  std::vector<double> label(num_categories, 0);
  label.front() = 1;

  // Evaluate network.
  std::vector<double> input_gradient;
  std::vector<double> param_gradient;
  const double loss = network.Evaluate(input, label, parameters,
                                       &input_gradient, &param_gradient);

  // Perturbation.
  const double delta = 1e-6;
  const double tol = 1e-6;

  // Estimate input gradient numerically.
  for (int i = 0; i < input.size(); ++i) {
    std::vector<double> input_delta = input;
    input_delta.at(i) += delta;

    std::vector<double> input_gradient_delta;
    std::vector<double> param_gradient_delta;

    const double loss_delta =
        network.Evaluate(input_delta, label, parameters, &input_gradient_delta,
                         &param_gradient_delta);

    const double numerical_gradient = (loss_delta - loss) / delta;
    if (std::abs(numerical_gradient - input_gradient.at(i)) > tol) {
      std::cerr << "Incorrect input gradient:"
                << " Analytical gradient: " << input_gradient.at(i)
                << " Numerical gradient: " << (loss_delta - loss) / delta
                << std::endl;
    }
  }

  // Estimate weight gradient numerically.
  for (int i = 0; i < parameters.size(); ++i) {
    std::vector<double> parameters_delta = parameters;
    parameters_delta.at(i) += delta;

    std::vector<double> input_gradient_delta;
    std::vector<double> param_gradient_delta;

    const double loss_delta =
        network.Evaluate(input, label, parameters_delta, &input_gradient_delta,
                         &param_gradient_delta);

    const double numerical_gradient = (loss_delta - loss) / delta;
    if (std::abs(numerical_gradient - param_gradient.at(i)) > tol) {
      std::cerr << "Incorrect param gradient:"
                << " Analytical gradient: " << param_gradient.at(i)
                << " Numerical gradient: " << (loss_delta - loss) / delta
                << std::endl;
    }
  }
}

void RunNetworkLearningTest() {
  // Create network.
  const int input_size = 10;  // TODO: Enable non-square inputs.
  const int input_channels = 3;
  const int num_categories = 10;
  const Network network =
      BuildTestNetwork(input_channels, input_size, input_size, num_categories);

  // Get initial set of parameters.
  std::vector<double> parameters = network.GetRandomParameters();

  // Generate random input.
  // TODO: Consider scaling, centering, normalization of input.
  const std::vector<double> input =
      GetRandomVector(input_size * input_size * input_channels, 0, 1);

  // Generate arbitrary label.
  std::vector<double> label(num_categories, 0);
  label.front() = 1;

  std::vector<double> first_moment(parameters.size(), 0);
  std::vector<double> second_moment(parameters.size(), 0);
  for (int i = 0; i < 1000; ++i) {
    // Evaluate network.
    std::vector<double> input_gradient;
    std::vector<double> param_gradient;
    const double loss = network.Evaluate(input, label, parameters,
                                         &input_gradient, &param_gradient);

    std::cerr << "Loss: " << loss << std::endl;

    std::vector<double> updated_network_params;
    std::vector<double> updated_first_moment;
    std::vector<double> updated_second_moment;
    AdamOptimizer(parameters, param_gradient, first_moment, second_moment, i,
                  &updated_network_params, &updated_first_moment,
                  &updated_second_moment);

    parameters = updated_network_params;
    first_moment = updated_first_moment;
    second_moment = updated_second_moment;
  }
}

#include "network_test.hpp"

#include <iostream>

#include "mnist.hpp"
#include "network.hpp"
#include "nn.hpp"
#include "training.hpp"

static void EvaulateNetworkPerformance(
    const Network& network, const std::vector<double>& parameters,
    const std::vector<Eigen::VectorXd>& inputs,
    const std::vector<Eigen::VectorXd>& labels) {
  int num_correct = 0;
  int num_evaluate = 1000;

  // Randomly sample num_evaluate test imgaes.
  std::vector<std::size_t> eval_indices(num_evaluate);
  std::iota(eval_indices.begin(), eval_indices.end(), 0);
  std::random_shuffle(eval_indices.begin(), eval_indices.end());

  std::vector<int> histogram(10, 0);

  for (int i = 0; i < eval_indices.size(); ++i) {
    const std::size_t ind = eval_indices.at(i);
    const Eigen::VectorXd& input_vec = inputs.at(ind);
    const Eigen::VectorXd& label_vec = labels.at(ind);
    const std::vector<double> input(input_vec.data(),
                                    input_vec.data() + input_vec.size());
    const std::vector<double> label(label_vec.data(),
                                    label_vec.data() + label_vec.size());
    const std::vector<double> output =
        network.Evaluate(input, label, parameters);

    // Compute max element of output.
    const std::size_t predicted_index = std::distance(
        output.begin(), std::max_element(output.begin(), output.end()));
    const std::size_t label_index = std::distance(
        label.begin(), std::max_element(label.begin(), label.end()));

    assert(predicted_index >= 0);
    assert(predicted_index < 10);
    assert(label_index >= 0);
    assert(label_index < 10);

    if (predicted_index == label_index) {
      ++num_correct;
    }

    ++histogram.at(predicted_index);
  }

  const double pct = 100 * double(num_correct) / double(num_evaluate);
  std::cerr << "Correctly predicted " << pct << "%" << std::endl;
  std::cerr << "Histogram of predictions: ";
  for (int i = 0; i < histogram.size(); ++i) {
    std::cerr << histogram.at(i) << ", ";
  }
  std::cerr << std::endl;
}

static Network BuildTestNetwork(const int input_channels, const int input_rows,
                                const int input_cols, const int num_outputs) {
  // Specify conv layers.
  const int num_kernels = 10;
  const int kernel_size = 3;  // TODO: Enable non-square kernels.
  const int stride = 1;

  // Fully connected layer width.
  const int num_nodes = 50;

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
  NetworkTiming network_timing;
  const double loss =
      network.Evaluate(input, label, parameters, &input_gradient,
                       &param_gradient, &network_timing);

  // Perturbation.
  const double delta = 1e-6;
  const double tol = 1e-6;

  // Estimate input gradient numerically.
  for (int i = 0; i < input.size(); ++i) {
    std::vector<double> input_delta = input;
    input_delta.at(i) += delta;

    std::vector<double> input_gradient_delta;
    std::vector<double> param_gradient_delta;

    NetworkTiming network_timing_delta;
    const double loss_delta =
        network.Evaluate(input_delta, label, parameters, &input_gradient_delta,
                         &param_gradient_delta, &network_timing_delta);

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

    NetworkTiming network_timing_delta;
    const double loss_delta =
        network.Evaluate(input, label, parameters_delta, &input_gradient_delta,
                         &param_gradient_delta, &network_timing_delta);

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
    NetworkTiming network_timing;
    const double loss =
        network.Evaluate(input, label, parameters, &input_gradient,
                         &param_gradient, &network_timing);

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

void RunNetworkMnistTest() {
  const int num_training_images = 60000;
  const int num_test_images = 10000;
  // Load training data.
  std::vector<Eigen::VectorXd> training_images;
  std::vector<Eigen::VectorXd> training_labels;
  LoadMnist("../data/mnist_train.csv", num_training_images, &training_images,
            &training_labels);

  // Load test data.
  std::vector<Eigen::VectorXd> test_images;
  std::vector<Eigen::VectorXd> test_labels;
  LoadMnist("../data/mnist_test.csv", num_test_images, &test_images,
            &test_labels);

  std::cerr << "Loaded data" << std::endl;

  // Create network.
  const int input_size = 28;
  const int input_channels = 1;
  const int num_categories = 10;
  const Network network =
      BuildTestNetwork(input_channels, input_size, input_size, num_categories);

  // Get initial set of parameters.
  std::vector<double> parameters = network.GetRandomParameters();

  std::cerr << "Loaded network" << std::endl;

  const int batch_size = 100;
  const int num_batches = num_training_images / batch_size;

  // Get randomly ordered training image indices.
  std::vector<int> indices(num_training_images);
  std::iota(indices.begin(), indices.end(), 0);
  std::random_shuffle(indices.begin(), indices.end());

  std::vector<double> first_moment(parameters.size(), 0);
  std::vector<double> second_moment(parameters.size(), 0);

  for (int i = 0; i < num_batches; ++i) {
    std::vector<double> param_gradient_batch_sum(parameters.size(), 0);
    double loss_batch_sum = 0;
    for (int j = 0; j < batch_size; ++j) {
      const int index = indices.at(i * batch_size + j);
      std::cerr << "Training image: " << index << std::endl;
      const Eigen::VectorXd& training_image_vec = training_images.at(index);
      const std::vector<double> training_image(
          training_image_vec.data(),
          training_image_vec.data() + training_image_vec.size());
      const Eigen::VectorXd& training_label_vec = training_labels.at(index);
      const std::vector<double> training_label(
          training_label_vec.data(),
          training_label_vec.data() + training_label_vec.size());

      // Evaluate network.
      std::vector<double> input_gradient;
      std::vector<double> param_gradient;
      NetworkTiming network_timing;
      const double loss =
          network.Evaluate(training_image, training_label, parameters,
                           &input_gradient, &param_gradient, &network_timing);

      std::cerr << "Forward: " << network_timing.forward_pass
                << "s, Backward: " << network_timing.backward_pass << "s"
                << std::endl;

      loss_batch_sum += loss;

      // Add gradient to batch sum.
      std::transform(param_gradient_batch_sum.begin(),
                     param_gradient_batch_sum.end(), param_gradient.begin(),
                     param_gradient_batch_sum.begin(), std::plus<double>());
    }

    const double loss_batch_mean = loss_batch_sum / batch_size;
    std::cerr << "Loss (batch mean): " << loss_batch_mean << std::endl;

    // Compute batch mean gradient.
    std::vector<double> param_gradient_batch_mean(parameters.size());
    std::transform(
        param_gradient_batch_sum.begin(), param_gradient_batch_sum.end(),
        param_gradient_batch_mean.begin(),
        [](const double& batch_sum) { return batch_sum / batch_size; });

    std::vector<double> updated_network_params;
    std::vector<double> updated_first_moment;
    std::vector<double> updated_second_moment;
    AdamOptimizer(parameters, param_gradient_batch_mean, first_moment,
                  second_moment, i, &updated_network_params,
                  &updated_first_moment, &updated_second_moment);

    parameters = updated_network_params;
    first_moment = updated_first_moment;
    second_moment = updated_second_moment;

    // Evaluate performance.
    std::cerr << "Evaluating..." << std::endl;
    EvaulateNetworkPerformance(network, parameters, test_images, test_labels);
  }
}

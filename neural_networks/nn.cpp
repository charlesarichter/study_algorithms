#include "nn.hpp"

#include <iostream>

double GetWeightCoefficient(const ActivationFunction& activation_function) {
  switch (activation_function) {
    case ActivationFunction::RELU: {
      return 0.001;
    }
    case ActivationFunction::SIGMOID: {
      return 0.1;
    }
    default: {
      // TODO: Fill in other activation functions.
      return 1.0;
    }
  }
}

std::vector<double> GetRandomVector(const int num_elements,
                                    const double min_value,
                                    const double max_value) {
  std::vector<double> random_vector(num_elements);
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(min_value, max_value);
  std::generate(
      random_vector.begin(), random_vector.end(),
      [&generator, &distribution]() { return distribution(generator); });
  return random_vector;
}

NeuralNetworkParameters GetRandomNeuralNetwork(
    int input_dimension, int output_dimension, int num_hidden_layers,
    int nodes_per_hidden_layer, const ActivationFunction hidden_activation,
    const ActivationFunction output_activation) {
  NeuralNetworkParameters nn;
  // Input layer
  nn.weights.emplace_back(
      Eigen::MatrixXd::Random(nodes_per_hidden_layer, input_dimension));
  nn.biases.emplace_back(Eigen::VectorXd::Random(nodes_per_hidden_layer));
  nn.activation_functions.emplace_back(hidden_activation);

  // Randomly initialize weights and biases for each layer
  for (int i = 0; i < num_hidden_layers - 1; ++i) {
    // Scale random weights based on the type of activation.
    const double weight_coefficient = GetWeightCoefficient(hidden_activation);
    nn.weights.emplace_back(weight_coefficient *
                            Eigen::MatrixXd::Random(nodes_per_hidden_layer,
                                                    nodes_per_hidden_layer));
    nn.biases.emplace_back(weight_coefficient *
                           Eigen::VectorXd::Random(nodes_per_hidden_layer));
    nn.activation_functions.emplace_back(hidden_activation);
  }

  // Output layer
  nn.weights.emplace_back(
      Eigen::MatrixXd::Random(output_dimension, nodes_per_hidden_layer));
  nn.biases.emplace_back(Eigen::VectorXd::Random(output_dimension));
  nn.activation_functions.emplace_back(output_activation);
  return nn;
}

// Example derivation of gradients:
//
// Consider a single hidden layer
// y = f1(A1*f0(A0*x0 + b0) + b1)
//
// A0 and A1 are weight matrices
// b0 and b1 are bias vectors
//
// want derivative of y with respect to A0, A1, b0, b1, *evaluated at x0*
//
// dy/dA1 = f1'(A1*f0(A0*x0 + b0) + b1) * f0(A0*x0 + b0)
// dy/db1 = f1'(A1*f0(A0*x0 + b0) + b1)
// dy/dA0 = f1'(A1*f0(A0*x0 + b0) + b1) * A1 * f0'(A0*x0 + b0) * x0
// dy/db0 = f1'(A1*f0(A0*x0 + b0) + b1) * A1 * f0'(A0*x0 + b0)
//
// Written differently, where x1 = f0(A0*x0 + b0),
// dy/dA1 = f1'(A1*x1 + b1) * x1
// dy/db1 = f1'(A1*x1 + b1)
// dy/dA0 = f1'(A1*x1 + b1) * A1 * f0'(A0*x0 + b0) * x0
// dy/db0 = f1'(A1*x1 + b1) * A1 * f0'(A0*x0 + b0)

// Example derivation of gradients:
//
// Consider a single hidden layer with linear activations
// y = A1*(A0*x0 + b0) + b1
// dy/dA1 = A0*x0 + b0
// dy/db1 = I

void EvaluateNetwork(
    const Eigen::VectorXd& input, const NeuralNetworkParameters& params,
    Eigen::VectorXd* output,
    std::vector<std::vector<Eigen::MatrixXd>>* weight_gradients,
    std::vector<std::vector<Eigen::VectorXd>>* bias_gradients) {
  // Forward pass
  Eigen::VectorXd current_value = input;
  std::vector<Eigen::MatrixXd> activation_gradients;
  std::vector<Eigen::VectorXd> post_activation_results;
  post_activation_results.emplace_back(input);
  for (int i = 0; i < params.weights.size(); ++i) {
    // Compute pre-activation input.
    const Eigen::VectorXd pre_activation =
        params.weights.at(i) * current_value + params.biases.at(i);

    // Compute activation output.
    Eigen::MatrixXd activation_gradient;
    current_value =
        Activation(pre_activation, params.activation_functions.at(i),
                   &activation_gradient);
    post_activation_results.emplace_back(current_value);
    activation_gradients.emplace_back(activation_gradient);
  }
  *output = current_value;

  // Determine dimensionality of the output.
  const int output_dimension = params.weights.back().rows();

  // Allocate output.
  weight_gradients->resize(params.weights.size());
  bias_gradients->resize(params.weights.size());
  for (int i = 0; i < params.weights.size(); ++i) {
    weight_gradients->at(i).resize(output_dimension);
    bias_gradients->at(i).resize(output_dimension);
  }

  for (int j = 0; j < output_dimension; ++j) {
    // Backward pass
    Eigen::MatrixXd a = Eigen::MatrixXd::Ones(1, 1);

    for (int i = (params.weights.size() - 1); i >= 0; --i) {
      // If the network had only one output, we would not have to do this, but
      // to handle networks where we want to compute gradients w.r.t. multiple
      // outputs (e.g., intermediate layers or softmax output) we need to start
      // the backward pass with the appropriate subset of the final layer
      // activation gradients.
      //
      // TODO: This seems pretty inelegant. Figure out a way to handle this
      // using a better logical setup (perhaps start the recursive process
      // before the loop) or by increasing the dimensionality of relevant data
      // structures by one (Matrix->Tensor, Vector->Matrix, etc.) so we don't
      // have to loop over j. Looping over j is essentially equivalent to
      // thinking about output_dimension number of independent networks.
      Eigen::MatrixXd current_act_grad;
      if (i == (params.weights.size() - 1)) {
        current_act_grad = activation_gradients.at(i).row(j);
      } else {
        current_act_grad = activation_gradients.at(i);
      }

      const Eigen::MatrixXd dydb = (a * current_act_grad).transpose();
      const Eigen::MatrixXd dydw =
          dydb * post_activation_results.at(i).transpose();
      a = a * current_act_grad * params.weights.at(i);
      weight_gradients->at(i).at(j) = dydw;
      bias_gradients->at(i).at(j) = dydb;
    }
  }
}

void EvaluateNetworkLoss(const Eigen::VectorXd& input,
                         const NeuralNetworkParameters& params,
                         const Eigen::VectorXd& label,
                         const LossFunction& loss_function,
                         Eigen::VectorXd* loss,
                         std::vector<Eigen::MatrixXd>* weight_gradients,
                         std::vector<Eigen::VectorXd>* bias_gradients) {
  // Compute output and gradients of the network itself.
  Eigen::VectorXd network_output;
  std::vector<std::vector<Eigen::MatrixXd>> network_weight_gradients;
  std::vector<std::vector<Eigen::VectorXd>> network_bias_gradients;
  EvaluateNetwork(input, params, &network_output, &network_weight_gradients,
                  &network_bias_gradients);

  // Compute the loss.
  Eigen::VectorXd loss_gradient;
  *loss = Loss(network_output, label, loss_function, &loss_gradient);

  // Compute gradients of loss w.r.t. network params.  TODO(charlie-or): Make
  // this N-dimensional. In order to do that, you would have to have
  // multi-output support everywhere else in the code. Instead of the derivative
  // of a single output with respect to each weight in a layer (i.e., 2D matrix)
  // you would have to have the derivative of each output with respect to each
  // weight in a layer (i.e., a stack of 2D matrices, or a 3D tensor, or a 2D
  // matrix where each column could be wrapped into a matrix of gradients for a
  // particular output). For now, we just assume that we have a scalar output
  // and scalar loss function and assert that its dimension is 1.
  // for (size_t i = 0; i < network_weight_gradients.size(); ++i) {
  //   assert(loss_gradient.size() == 1);
  //   const double loss_gradient_1d = loss_gradient(0);
  //   weight_gradients->emplace_back(network_weight_gradients.at(i) *
  //                                  loss_gradient_1d);
  // }
  // for (size_t i = 0; i < network_bias_gradients.size(); ++i) {
  //   assert(loss_gradient.size() == 1);
  //   const double loss_gradient_1d = loss_gradient(0);
  //   bias_gradients->emplace_back(network_bias_gradients.at(i) *
  //                                loss_gradient_1d);
  // }

  // // Want: dloss/dweights
  // // Have: dloss/dppredicted, dppredicted/weights
  // // dloss/dweights = dloss/dpredicted * dpredicted/dweights
  for (size_t i = 0; i < network_weight_gradients.size(); ++i) {
    // TODO: Clean up variable names and indexing.
    Eigen::MatrixXd analytical_weight_gradient = Eigen::MatrixXd::Zero(
        params.weights.at(i).rows(), params.weights.at(i).cols());
    Eigen::VectorXd analytical_bias_gradient =
        Eigen::VectorXd::Zero(params.biases.at(i).size());

    for (size_t j = 0; j < network_output.size(); ++j) {
      analytical_weight_gradient +=
          network_weight_gradients.at(i).at(j) * loss_gradient(j);
      analytical_bias_gradient +=
          network_bias_gradients.at(i).at(j) * loss_gradient(j);
    }
    weight_gradients->emplace_back(analytical_weight_gradient);
    bias_gradients->emplace_back(analytical_bias_gradient);
  }
}

void EvaluateNetworkLossCombinedImplementation(
    const Eigen::VectorXd& input, const NeuralNetworkParameters& params,
    const Eigen::VectorXd& label, const LossFunction& loss_function,
    Eigen::VectorXd* loss, std::vector<Eigen::MatrixXd>* weight_gradients,
    std::vector<Eigen::VectorXd>* bias_gradients) {
  // Forward pass
  Eigen::VectorXd current_value = input;
  std::vector<Eigen::MatrixXd> activation_gradients;
  std::vector<Eigen::VectorXd> post_activation_results;
  post_activation_results.emplace_back(input);
  for (int i = 0; i < params.weights.size(); ++i) {
    // Compute pre-activation input.
    const Eigen::VectorXd pre_activation =
        params.weights.at(i) * current_value + params.biases.at(i);

    // Compute activation output.
    Eigen::MatrixXd activation_gradient;
    current_value =
        Activation(pre_activation, params.activation_functions.at(i),
                   &activation_gradient);
    post_activation_results.emplace_back(current_value);
    activation_gradients.emplace_back(activation_gradient);
  }

  // Compute loss
  Eigen::VectorXd loss_gradient;
  *loss = Loss(current_value, label, loss_function, &loss_gradient);

  // Allocate gradient output
  weight_gradients->resize(params.weights.size());
  bias_gradients->resize(params.weights.size());

  // Backward pass
  Eigen::MatrixXd a = loss_gradient.transpose();
  for (int i = (params.weights.size() - 1); i >= 0; --i) {
    Eigen::MatrixXd current_act_grad;
    current_act_grad = activation_gradients.at(i);
    const Eigen::MatrixXd dydb = (a * current_act_grad).transpose();
    const Eigen::MatrixXd dydw =
        dydb * post_activation_results.at(i).transpose();
    a = a * current_act_grad * params.weights.at(i);
    weight_gradients->at(i) = dydw;
    bias_gradients->at(i) = dydb;
  }
}

Eigen::VectorXd Activation(const Eigen::VectorXd& input,
                           const ActivationFunction activation_function,
                           Eigen::MatrixXd* activation_gradient) {
  Eigen::VectorXd output(input.size());
  switch (activation_function) {
    case ActivationFunction::LINEAR: {
      output = input;

      // Slope of one.
      *activation_gradient =
          Eigen::MatrixXd::Identity(input.size(), input.size());
      break;
    }
    case ActivationFunction::RELU: {
      // TODO: More efficient/vectorized computation.
      *activation_gradient = Eigen::MatrixXd::Zero(input.size(), input.size());
      for (size_t i = 0; i < input.size(); ++i) {
        if (input(i) <= 0) {
          output(i) = 0;
          (*activation_gradient)(i, i) = 0;
        } else {
          output(i) = input(i);
          (*activation_gradient)(i, i) = 1;
        }
      }
      break;
    }
    case ActivationFunction::SIGMOID: {
      // The "Sigmoid" (a.k.a. "Logistic") function squashes each element in
      // the input into the range (0,1). These outputs are independent of
      // one another. There is no normalization across all the outputs
      // resulting from a vector of inputs (unlike Softmax, which is
      // normalized across outputs).

      // TODO: More efficient/vectorized computation.
      *activation_gradient = Eigen::MatrixXd::Zero(input.size(), input.size());
      for (size_t i = 0; i < input.size(); ++i) {
        const double f = 1 / (1 + exp(-1 * input(i)));  // "Sigmoid"/"Logistic"
        output(i) = f;
        (*activation_gradient)(i, i) = f * (1 - f);
      }
      break;
    }
    case ActivationFunction::SOFTMAX: {
      // The Softmax function is a multi-dimensional generalization of the
      // logistic function. When the number of output dimensions is 2, the
      // softmax function is equivalent to the logistic function. One of the
      // outputs represents the probability that the coin flip result is
      // "heads" and the other output represents the probability that the
      // coin flip result is "tails" (i.e., the complement).
      Eigen::VectorXd result_unnormalized = Eigen::VectorXd::Zero(input.size());
      double activation_sum = 0;
      for (size_t i = 0; i < input.size(); ++i) {
        double activation_unnormalized = exp(input(i));
        result_unnormalized(i) = activation_unnormalized;
        activation_sum += activation_unnormalized;
      }
      // std::cerr << "softmax input " << input.transpose() << std::endl;
      // std::cerr << "softmax result unnormalized "
      //           << result_unnormalized.transpose() << std::endl;
      // std::cerr << "softmax activation sum " << activation_sum << std::endl;
      // std::cin.get();
      output = result_unnormalized / activation_sum;

      *activation_gradient = Eigen::MatrixXd::Zero(input.size(), input.size());
      for (size_t i = 0; i < input.size(); ++i) {
        for (size_t j = 0; j < input.size(); ++j) {
          const double kroneker_delta = (i == j) ? 1.0 : 0.0;
          (*activation_gradient)(i, j) =
              output(i) * (kroneker_delta - output(j));
        }
      }

      // Gradient of softmax: f_i(x)*(kroneker_delta_ij - f_j(x))
      // where f_i(x) = exp(x_i) / sum_j^N(exp(x_j)
      //
      // Kroneker delta: 1 when i = j and 0 when i != j.
      //
      // NOTE: The softmax gradient will be a matrix. Also, note that for
      // the other activation functions implemented so far, that gradient
      // matrix is diagonal, so it's possible that there is a bug throughout
      // all of the code requiring all gradient matrices to be transposed to
      // be correct for dense, non-symmetric gradient matrices. It would be
      // a good idea to run a test to find that out.

      break;
    }
    default: {
      throw std::runtime_error("Invalid activation type.");
      break;
    }
  }
  return output;
}

std::vector<double> Activation(const std::vector<double>& input,
                               const ActivationFunction activation_function,
                               std::vector<double>* activation_gradient) {
  std::vector<double> output(input.size());
  switch (activation_function) {
    case ActivationFunction::LINEAR: {
      output = input;

      // Slope of one.
      // Construct identity matrix elementwise.
      *activation_gradient =
          std::vector<double>(input.size() * input.size(), 0);
      for (size_t i = 0; i < input.size(); ++i) {
        const std::size_t ind = i + i * input.size();
        activation_gradient->at(ind) = 1;
      }

      break;
    }
    case ActivationFunction::RELU: {
      *activation_gradient =
          std::vector<double>(input.size() * input.size(), 0);
      for (size_t i = 0; i < input.size(); ++i) {
        const std::size_t ind = i + i * input.size();
        if (input.at(i) <= 0) {
          output.at(i) = 0;
          // Don't need to set this since gradient is zero initialized.
          // activation_gradient->at(ind) = 0;
        } else {
          output.at(i) = input.at(i);
          activation_gradient->at(ind) = 1;
        }
      }
      break;
    }
    case ActivationFunction::SIGMOID: {
      // The "Sigmoid" (a.k.a. "Logistic") function squashes each element in
      // the input into the range (0,1). These outputs are independent of
      // one another. There is no normalization across all the outputs
      // resulting from a vector of inputs (unlike Softmax, which is
      // normalized across outputs).
      *activation_gradient =
          std::vector<double>(input.size() * input.size(), 0);
      for (size_t i = 0; i < input.size(); ++i) {
        const std::size_t ind = i + i * input.size();
        const double f = 1 / (1 + exp(-1 * input.at(i)));
        output.at(i) = f;
        activation_gradient->at(ind) = f * (1 - f);
      }
      break;
    }
    case ActivationFunction::SOFTMAX: {
      // The Softmax function is a multi-dimensional generalization of the
      // logistic function. When the number of output dimensions is 2, the
      // softmax function is equivalent to the logistic function. One of the
      // outputs represents the probability that the coin flip result is
      // "heads" and the other output represents the probability that the
      // coin flip result is "tails" (i.e., the complement).
      std::vector<double> result_unnormalized(input.size(), 0);
      double activation_sum = 0;
      for (size_t i = 0; i < input.size(); ++i) {
        double activation_unnormalized = exp(input.at(i));
        result_unnormalized.at(i) = activation_unnormalized;
        activation_sum += activation_unnormalized;
      }
      std::transform(result_unnormalized.begin(), result_unnormalized.end(),
                     output.begin(), [activation_sum](const double& r) {
                       return r / activation_sum;
                     });
      *activation_gradient =
          std::vector<double>(input.size() * input.size(), 0);
      for (size_t i = 0; i < input.size(); ++i) {
        for (size_t j = 0; j < input.size(); ++j) {
          const std::size_t ind = j + i * input.size();

          const double kroneker_delta = (i == j) ? 1.0 : 0.0;
          activation_gradient->at(ind) =
              output.at(i) * (kroneker_delta - output.at(j));
        }
      }
      // Gradient of softmax: f_i(x)*(kroneker_delta_ij - f_j(x))
      // where f_i(x) = exp(x_i) / sum_j^N(exp(x_j)
      //
      // Kroneker delta: 1 when i = j and 0 when i != j.
      //
      // NOTE: The softmax gradient will be a matrix. Also, note that for
      // the other activation functions implemented so far, that gradient
      // matrix is diagonal, so it's possible that there is a bug throughout
      // all of the code requiring all gradient matrices to be transposed to
      // be correct for dense, non-symmetric gradient matrices. It would be
      // a good idea to run a test to find that out.
      break;
    }
    default: {
      throw std::runtime_error("Invalid activation type.");
      break;
    }
  }
  return output;
}

Eigen::VectorXd Loss(const Eigen::VectorXd& input, const Eigen::VectorXd& label,
                     const LossFunction loss_function,
                     Eigen::VectorXd* loss_gradient) {
  assert(input.size() == label.size());
  switch (loss_function) {
    case LossFunction::CROSS_ENTROPY: {
      // NOTE: Different definitions of cross entropy. Which do we want? See
      // this post: https://datascience.stackexchange.com/questions/9302/
      //            the-cross-entropy-error-function-in-neural-networks
      if (input.size() == 1) {
        // Binary cross-entropy loss not supported at the moment. This was
        // initially implemented, but then moved to implement/support
        // multi-class cross-entropy instead (for binary classification
        // problems, you can just use a 2-dimensional label (1,0) or (0,1)).
        //
        // assert(label.size() == 1);
        // const double label_1d = label(0);
        // const double p_predicted_1d = input(0);
        // const double loss_1d = -1 * (label_1d * log(p_predicted_1d) +
        //                              (1 - label_1d) * log(1 -
        //                              p_predicted_1d));
        // const Eigen::VectorXd loss = loss_1d * Eigen::VectorXd::Ones(1);
        //
        // // Compute gradients.
        // const double dloss_dpredicted_1d =
        //     -label_1d * (1 / p_predicted_1d) -
        //     (label_1d - 1) / (1 - p_predicted_1d);
        // *loss_gradient = dloss_dpredicted_1d * Eigen::VectorXd::Ones(1);
        // return loss;
        // break;
        throw std::runtime_error(
            "Binary cross-entropy loss not currently supported. Try multiclass "
            "cross-entropy instead.");
      } else {
        assert(input.size() > 1);

        double loss = 0;
        *loss_gradient = Eigen::VectorXd::Zero(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
          const double label_1d = label(i);
          const double p_predicted_1d = input(i);
          loss += -1 * (label_1d * log(p_predicted_1d) +
                        (1 - label_1d) * log(1 - p_predicted_1d));
          (*loss_gradient)(i) = -label_1d * (1 / p_predicted_1d) -
                                (label_1d - 1) / (1 - p_predicted_1d);
        }
        const Eigen::VectorXd loss_vec = loss * Eigen::VectorXd::Ones(1);

        // Compute gradients.
        // const double dloss_dpredicted_1d =
        //     -label_1d * (1 / p_predicted_1d) -
        //     (label_1d - 1) / (1 - p_predicted_1d);
        // *loss_gradient = dloss_dpredicted_1d * Eigen::VectorXd::Ones(1);

        return loss_vec;
        break;
      }
    }
    default: {
      throw std::runtime_error("Unsupported loss function");
      break;
    }
  }
}

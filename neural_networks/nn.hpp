#include <eigen3/Eigen/Dense>
#include <vector>

// NOTE: See https://en.wikipedia.org/wiki/activation_function
enum class ActivationFunction { LINEAR, RELU, SIGMOID, SOFTMAX };

enum class LossFunction { SQUARED_ERROR, CROSS_ENTROPY };

struct NeuralNetworkParameters {
  std::vector<Eigen::MatrixXd> weights;
  std::vector<Eigen::VectorXd> biases;
  std::vector<ActivationFunction> activation_functions;
};

NeuralNetworkParameters GetRandomNeuralNetwork(
    int input_dimension, int output_dimension, int num_hidden_layers,
    int nodes_per_hidden_layer, const ActivationFunction hidden_activation,
    const ActivationFunction output_activation);

void EvaluateNetwork(const Eigen::VectorXd& input,
                     const NeuralNetworkParameters& params,
                     Eigen::VectorXd* output,
                     std::vector<Eigen::MatrixXd>* weight_gradients,
                     std::vector<Eigen::VectorXd>* bias_gradients);

void EvaluateNetworkLoss(const Eigen::VectorXd& input,
                         const NeuralNetworkParameters& params,
                         const Eigen::VectorXd& label,
                         const LossFunction& loss_function,
                         Eigen::VectorXd* loss,
                         std::vector<Eigen::MatrixXd>* weight_gradients,
                         std::vector<Eigen::VectorXd>* bias_gradients);

Eigen::VectorXd Activation(const Eigen::VectorXd& input,
                           const ActivationFunction activation_function,
                           Eigen::MatrixXd* activation_gradient);

Eigen::VectorXd Loss(const Eigen::VectorXd& input, const Eigen::VectorXd& label,
                     const LossFunction loss_function,
                     Eigen::VectorXd* loss_gradient);

#include <eigen3/Eigen/Dense>
#include <vector>

// NOTE: See https://en.wikipedia.org/wiki/activation_function
enum class ActivationFunction { LINEAR, SIGMOID, RELU };

struct NeuralNetworkParameters {
  std::vector<Eigen::MatrixXd> weights;
  std::vector<Eigen::VectorXd> biases;
  std::vector<ActivationFunction> activation_functions;
};

// struct NeuralNetworkJacobians {};

NeuralNetworkParameters GetRandomNeuralNetwork(
    int input_dimension, int output_dimension, int num_hidden_layers,
    int nodes_per_hidden_layer, const ActivationFunction hidden_activation,
    const ActivationFunction output_activation);

void EvaluateNetwork(const Eigen::VectorXd& input,
                     const NeuralNetworkParameters& params,
                     Eigen::VectorXd* output,
                     std::vector<Eigen::MatrixXd>* weight_gradients);

Eigen::VectorXd Activation(const Eigen::VectorXd& input,
                           const ActivationFunction activation_function,
                           Eigen::VectorXd* activation_gradient);

// class NeuralNetwork {
//  public:
//   NeuralNetwork(const int input_dimension, const int output_dimension,
//                 const int num_hidden_layers, const int
//                 nodes_per_hidden_layer, const ActivationFunction
//                 hidden_activation, const ActivationFunction
//                 output_activation);
//
//   ~NeuralNetwork();
//
//   void EvaluateNetwork(const Eigen::VectorXd& input, Eigen::VectorXd* output,
//                        std::vector<Eigen::MatrixXd>* weight_jacobian,
//                        std::vector<Eigen::VectorXd>* bias_jacobian);
//   Eigen::VectorXd Activation(const Eigen::VectorXd& input,
//                              const ActivationFunction activation_function);
//
//  private:
//   std::vector<Eigen::MatrixXd> weights_;
//   std::vector<Eigen::VectorXd> biases_;
//   std::vector<ActivationFunction> activation_functions_;
// }
// ;

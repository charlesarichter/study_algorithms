#pragma once

#include <eigen3/Eigen/Dense>
#include <memory>
#include <vector>

#include "conv_structs.hpp"
#include "nn.hpp"

class Layer;
class LayerFC;
class LayerConv;

using LayerPtr = std::shared_ptr<Layer>;
using LayerFCPtr = std::shared_ptr<LayerFC>;
using LayerConvPtr = std::shared_ptr<LayerConv>;

enum class LayerType { FC, CONV };

class Layer {
 public:
  virtual LayerType GetLayerType() const = 0;

  virtual int GetNumParameters() const = 0;

  virtual std::vector<double> GetRandomParameters() const = 0;

  virtual void ForwardPass(const std::vector<double>& input,
                           const std::vector<double>& parameters,
                           std::vector<double>* output,
                           std::vector<double>* activation_gradient) const = 0;

  // virtual InputOutputVolume ForwardPass(const InputOutputVolume& input) = 0;

  // virtual InputOutputVolume BackwardPass(const InputOutputVolume& input) = 0;

  // TODO: virtual WeightGradient() = 0;

  // virtual void SetParameters(const std::vector<double>& parameters) = 0;
};

class LayerFC : public Layer {
 public:
  LayerFC(const int num_inputs, const int num_outputs,
          const ActivationFunction& activation_function)
      : num_inputs_(num_inputs),
        num_outputs_(num_outputs),
        activation_function_(activation_function) {}

  LayerType GetLayerType() const { return LayerType::FC; }

  int GetNumParameters() const;

  std::vector<double> GetRandomParameters() const;

  void ForwardPass(const std::vector<double>& input,
                   const std::vector<double>& parameters,
                   std::vector<double>* output,
                   std::vector<double>* activation_gradient) const;

 private:
  int num_inputs_;
  int num_outputs_;

  ActivationFunction activation_function_;
};

class LayerConv : public Layer {
 public:
  LayerConv(const int input_rows, const int input_cols,
            const int input_channels, const int kernel_rows,
            const int kernel_cols, const int num_kernels, const int stride,
            const ActivationFunction& activation_function)
      : input_rows_(input_rows),
        input_cols_(input_cols),
        input_channels_(input_channels),
        kernel_rows_(kernel_rows),
        kernel_cols_(kernel_cols),
        num_kernels_(num_kernels),
        stride_(stride),
        activation_function_(activation_function) {}

  LayerType GetLayerType() const { return LayerType::CONV; }

  int GetNumParameters() const;

  std::vector<double> GetRandomParameters() const;

  void ForwardPass(const std::vector<double>& input,
                   const std::vector<double>& parameters,
                   std::vector<double>* output,
                   std::vector<double>* activation_gradient) const;

  int GetOutputRows() const {
    return (input_rows_ - kernel_rows_) / stride_ + 1;
  }
  int GetOutputCols() const {
    return (input_cols_ - kernel_cols_) / stride_ + 1;
  }
  int GetOutputChannels() const { return num_kernels_; }

 private:
  int input_rows_;
  int input_cols_;
  int input_channels_;
  int kernel_rows_;
  int kernel_cols_;
  int num_kernels_;
  int stride_;

  // TODO: Set this value in constructor.
  int padding_{0};

  ActivationFunction activation_function_;
};

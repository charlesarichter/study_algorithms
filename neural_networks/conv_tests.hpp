#pragma once

#include "conv_example.hpp"
#include "conv_structs.hpp"

void RunConvTests();
void RunConvGradientTests();
void RunConvKernelTests();
void TestConv(const ConvExample& conv_example);
void TestConvGradient(const ConvExample& conv_example);
void TestConvKernels(const ConvExample& conv_example);

#cmake_minimum_required (VERSION 2.6)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")

# This line helps the unsupported eigen FFT module find Eigen/Core
# See: https://codeyarns.github.io/tech/2013-10-10-how-to-add-include-directories-in-cmake.html
include_directories(/usr/include/eigen3)

add_library(nn
 nn.cpp
 nn.hpp
 conv.cpp
 conv.hpp
 conv_structs.cpp
 conv_structs.hpp
 conv_example.cpp
 conv_example.hpp
 conv_tests.cpp
 conv_tests.hpp
 layer.cpp
 layer.hpp
 network.cpp
 network.hpp
 network_test.cpp
 network_test.hpp
 training.cpp
 training.hpp
 mnist.cpp
 mnist.hpp)

add_executable(nn_app main.cpp)
target_link_libraries(nn_app nn)

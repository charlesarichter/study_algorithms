#cmake_minimum_required (VERSION 2.6)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

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

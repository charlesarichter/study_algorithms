#cmake_minimum_required (VERSION 2.6)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")

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
 training.cpp
 training.hpp)

add_executable(nn_app main.cpp)
target_link_libraries(nn_app nn)

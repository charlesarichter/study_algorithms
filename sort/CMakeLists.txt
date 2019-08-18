#cmake_minimum_required (VERSION 2.6)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")

add_executable(sort_app main.cpp sort.hpp sort.cpp)

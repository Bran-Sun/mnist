cmake_minimum_required(VERSION 3.6)
project(mnist)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

set(SOURCE_FILES main.cpp layer.cpp layer.h function.cpp function.h network.cpp network.h data.cpp data.h mnist_loader.cpp mnist_loader.h)
add_executable(mnist ${SOURCE_FILES})
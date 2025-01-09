cmake_minimum_required(VERSION 3.30)
project(NN_train)

set(CMAKE_CXX_STANDARD 20)

add_executable(NN_train main.cpp ../lib/NN.cpp ../lib/NN.h ../lib/readData.cpp ../lib/readData.h)
file(COPY ${CMAKE_SOURCE_DIR}/Data DESTINATION ${CMAKE_BINARY_DIR})
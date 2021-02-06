cmake_minimum_required(VERSION 3.10)
project(knnsearch)

set(CMAKE_CXX_STANDARD 17)

set(INC_DIR ../knnsearch/include)
set(LINK_DIR ../knnsearch/src)

include_directories(${INC_DIR})
link_directories(${LINK_DIR})

set(SOURCE_FILES test.cpp)

add_executable(knnsearch test.cpp include/kdTree.h src/kdTree.cpp)
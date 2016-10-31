cmake_minimum_required(VERSION 3.6)
project(VvsE)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES
    src/data/mod.c
    src/src/graph.c
    src/src/graph.h
    src/src/partition.c
    src/src/partition.h
    src/src/timer.h
    src/bfs.cu
    src/cc.cu
    src/main.cu
    src/Makefile
    src/pr.cu
    src/sssp.cu
    LICENSE
    README.md)

add_executable(VvsE ${SOURCE_FILES})
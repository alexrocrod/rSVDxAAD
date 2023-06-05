cmake_minimum_required(VERSION 3.10)
project(rSVDxAAD_rSVD)    
include(../options.cmake)

add_executable(svd3 svd3.cpp)
SET_TARGET_PROPERTIES(svd3 PROPERTIES COMPILE_FLAGS "-fopenmp" LINK_FLAGS "-fopenmp")
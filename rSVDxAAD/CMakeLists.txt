cmake_minimum_required(VERSION 3.10)
project(rSVDxAAD_rSVDxAAD)    
include(../options.cmake)

add_library(SVDxAdjHelper INTERFACE)

add_executable(rSVDxAAD rSVDxAAD.cpp)
SET_TARGET_PROPERTIES(rSVDxAAD PROPERTIES COMPILE_FLAGS "-fopenmp" LINK_FLAGS "-fopenmp")
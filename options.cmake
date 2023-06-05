set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set (CMAKE_CXX_STANDARD 14)

message("CMAKE_CURRENT_LIST_DIR is ${CMAKE_CURRENT_LIST_DIR}")

include_directories(${CMAKE_CURRENT_LIST_DIR}/3rdparty)
include_directories(${CMAKE_CURRENT_LIST_DIR}/3rdparty/Eigen)

# find_package(Threads)
find_package(OpenMP)
# if(OpenMP_CXX_FOUND)
#     link_libraries(aadc OpenMP::OpenMP_CXX)
# endif()

# if(OpenMP_CXX_FOUND)
#     target_link_libraries(MyTarget PUBLIC OpenMP::OpenMP_CXX)
# endif()

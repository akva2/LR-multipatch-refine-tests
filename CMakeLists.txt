project(LR-multipatch-refine)

cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(LRSpline REQUIRED)

include_directories(${LRSpline_INCLUDE_DIRS})

add_executable(TestMatchingKnots src/TestMatchingKnots.cpp)
target_link_libraries(TestMatchingKnots ${LRSpline_LIBRARIES})

enable_testing()
add_test(2patch_flip+reverse TestMatchingKnots 1)

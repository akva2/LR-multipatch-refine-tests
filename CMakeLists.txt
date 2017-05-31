project(LR-multipatch-refine)

cmake_minimum_required(VERSION 3.2)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(USE_MPI "Compile with MPI support?" OFF)

find_package(LRSpline REQUIRED)
find_package(GoTools  REQUIRED)
if(USE_MPI)
  find_package(MPI REQUIRED)
endif()

include_directories(${LRSpline_INCLUDE_DIRS} ${GoTools_INCLUDE_DIRS} ${MPI_INCLUDE_DIRS})
add_definitions(${LRSpline_DEFINITIONS})

add_executable(TestMatchingKnots src/TestMatchingKnots.cpp)
target_link_libraries(TestMatchingKnots ${LRSpline_LIBRARIES} ${GoTools_LIBRARIES} ${MPI_LIBRARIES})

enable_testing()
if(USE_MPI)
  add_test(2patch_flip+reverse ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} TestMatchingKnots ${MPIEXEC_POSTFLAGS} 1)
else()
  add_test(2patch_flip+reverse TestMatchingKnots 1)
endif()

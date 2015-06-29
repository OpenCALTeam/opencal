# opencal/OpenCAL-OMP/examples/sciddicaT/CMakeLists.txt
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/source)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

LINK_DIRECTORIES(${PROJECT_SOURCE_DIR}/lib)

file(GLOB SRC
    "source/*.c"
)

add_executable(calomp_sciddicaT ${SRC})
target_link_libraries(calomp_sciddicaT opencal-omp ${CMAKE_THREAD_LIBS_INIT})

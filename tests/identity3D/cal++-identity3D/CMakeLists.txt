# opencal/OpenCAL/examples/cal_life/CMakeLists.txt
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/OpenCAL++/include)

file(GLOB SRC
    "include/*.h"
    "source/*.cpp"
    "source/*.c"
)

link_directories(${PROJECT_SOURCE_DIR}/OpenCAL++/lib)

add_executable(cal++-identity3D-test ${SRC})
set_property(TARGET cal++-identity3D-test PROPERTY CXX_STANDARD 14)

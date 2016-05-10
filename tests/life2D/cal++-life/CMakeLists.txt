cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/OpenCAL++/include)

file(GLOB SRC
    "include/*.h"
    "source/*.cpp"
)

link_directories(${PROJECT_SOURCE_DIR}/OpenCAL++/lib)
add_executable(cal++-life-test ${SRC})
target_link_libraries(cal++-life-test opencal++)
set_property(TARGET cal++-life-test PROPERTY CXX_STANDARD 14)

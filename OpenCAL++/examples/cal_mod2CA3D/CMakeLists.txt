# opencal/OpenCAL/examples/cal_mod2CA3D/CMakeLists.txt
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/source)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

file(GLOB SRC
    "source/*.cpp"
)

add_executable(calpp_mod2CA3D ${SRC})

target_link_libraries(calpp_mod2CA3D opencalpp)


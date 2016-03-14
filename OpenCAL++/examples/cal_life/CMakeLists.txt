# opencal/OpenCAL/examples/cal_life/CMakeLists.txt
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/source)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

file(GLOB SRC
    "include/*.h"
    "source/*.cpp"
)

add_executable(calpp_life ${SRC})
<<<<<<< HEAD
<<<<<<< HEAD
target_link_libraries(calpp_life opencalpp )
=======
target_link_libraries(calpp_life opencalpp11 )
>>>>>>> e44630b317eeb506eac14bb3076f71487fe5ed2d
=======
target_link_libraries(calpp_life opencalpp11 )
>>>>>>> e44630b317eeb506eac14bb3076f71487fe5ed2d

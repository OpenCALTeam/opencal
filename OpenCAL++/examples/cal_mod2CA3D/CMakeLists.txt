# opencal/OpenCAL/examples/cal_mod2CA3D/CMakeLists.txt
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/source)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

file(GLOB SRC
    "source/*.cpp"
)

add_executable(calpp_mod2CA3D ${SRC})
<<<<<<< HEAD
<<<<<<< HEAD
target_link_libraries(calpp_mod2CA3D opencalpp)
=======
target_link_libraries(calpp_mod2CA3D opencalpp11)
>>>>>>> e44630b317eeb506eac14bb3076f71487fe5ed2d
=======
target_link_libraries(calpp_mod2CA3D opencalpp11)
>>>>>>> e44630b317eeb506eac14bb3076f71487fe5ed2d

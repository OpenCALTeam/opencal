cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

OPTION( EXAMPLES "EXAMPLES" OFF)

add_subdirectory(OpenCAL)
add_subdirectory(OpenCAL-CL)
add_subdirectory(OpenCAL-GL)
add_subdirectory(OpenCAL-OMP)

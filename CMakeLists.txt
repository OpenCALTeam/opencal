# opencal/CMakeLists.txt

cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

project(OpenCAL)

OPTION(EXAMPLES "EXAMPLES" OFF)
OPTION(FREEGLUT_STATIC "Use static lib" OFF)
OPTION(PTW32_STATIC "Use pthread Win32 static lib" OFF)

add_subdirectory(OpenCAL)
add_subdirectory(OpenCAL-CL)
add_subdirectory(OpenCAL-GL)
add_subdirectory(OpenCAL-OMP)

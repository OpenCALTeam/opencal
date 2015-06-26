# opencal/CMakeLists.txt

cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(OperCAL)

OPTION(EXAMPLES "EXAMPLES" OFF)
OPTION(FREEGLUT_STATIC "Use static lib" OFF)
OPTION(PTW32_STATIC "Use pthread Win32 static lib" OFF)
OPTION(BUILD_OPENCL "Build the OpenCL version: (Opencl required)" ON)
OPTION(BUILD_GL "Build the OpenCL version: (Opencl required)" ON)
OPTION(BUILD_OMP "Build the OpenCL version: (Opencl required)" ON)
option(BUILD_DOCUMENTATION "Create and install the HTML based API documentation (requires Doxygen)" OFF)

if(BUILD_DOCUMENTATION)
	find_package(Doxygen)
endif(BUILD_DOCUMENTATION)

add_subdirectory(OpenCAL)

if(BUILD_OPENCL)
	add_subdirectory(OpenCAL-CL)
endif(BUILD_OPENCL)

if(BUILD_GL)
	add_subdirectory(OpenCAL-GL)
endif(BUILD_GL)

if(BUILD_OMP)
	add_subdirectory(OpenCAL-OMP)
endif(BUILD_OMP)

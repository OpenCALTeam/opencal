# opencal/CMakeLists.txt
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(OpenCAL-ALL)

option(BUILD_DOCUMENTATION  "Build the HTML based API documentation (Doxygen required)"             OFF)
option(BUILD_OPENCAL_SERIAL "Build the OpenCAL serial version"                                      ON )
option(BUILD_OPENCAL_OMP    "Build the OpenCAL-OMP OpenMP parallel version (OpenMP required)"       OFF)
option(BUILD_OPENCAL_CL     "Build the OpenCAL-CL OpenCL parallel version (OpenCL required)"        OFF)
option(BUILD_OPENCAL_GL     "Build the OpenCAL-GL visualization library (OpenGL and GLUT required)" OFF)
option(BUILD_EXAMPLES       "Build the examples for each OpenCAL version"                           OFF)
option(BUILD_OPENCAL_PP     "Build the OpenCAL-C++ version (C++11 Required)"		 				OFF)

if(BUILD_DOCUMENTATION)
	find_package(Doxygen)
endif(BUILD_DOCUMENTATION)

if (BUILD_OPENCAL_SERIAL)
	add_subdirectory(OpenCAL)
endif(BUILD_OPENCAL_SERIAL)

if(BUILD_OPENCAL_OMP)
	add_subdirectory(OpenCAL-OMP)
endif(BUILD_OPENCAL_OMP)

if(BUILD_OPENCAL_CL)
	add_subdirectory(OpenCAL-CL)
endif(BUILD_OPENCAL_CL)

if(BUILD_OPENCAL_GL)
	add_subdirectory(OpenCAL-GL)
endif(BUILD_OPENCAL_GL)

if(BUILD_OPENCAL_PP)
	add_subdirectory(OpenCAL++)
endif(BUILD_OPENCAL_PP)


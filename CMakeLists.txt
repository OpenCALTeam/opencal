# opencal/CMakeLists.txt
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
cmake_policy(SET CMP0048 NEW)
project(OpenCAL-ALL 
		VERSION 0.1
		LANGUAGES C CXX)

option(BUILD_DOCUMENTATION  "Build the HTML based API documentation (Doxygen required)"             OFF)
option(BUILD_OPENCAL_SERIAL "Build the OpenCAL serial version"                                      ON )
option(BUILD_OPENCAL_OMP    "Build the OpenCAL-OMP OpenMP parallel version (OpenMP required)"       OFF)
option(BUILD_OPENCAL_CL     "Build the OpenCAL-CL OpenCL parallel version (OpenCL required)"        OFF)
option(BUILD_OPENCAL_GL     "Build the OpenCAL-GL visualization library (OpenGL and GLUT required)" OFF)
option(BUILD_EXAMPLES       "Build the examples for each OpenCAL version"                           OFF)
option(BUILD_OPENCAL_PP     "Build the OpenCAL-C++ version (C++11 Required)"		 				OFF)
option(BUILD_OPENCAL_TESTS  "Build the test suite of application for OPENCAL"		 				OFF)

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

if(BUILD_OPENCAL_TESTS)
	add_subdirectory(tests)
endif(BUILD_OPENCAL_TESTS)

# uninstall target
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
    IMMEDIATE @ONLY)

add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)

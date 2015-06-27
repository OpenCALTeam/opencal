# opencal/CMakeLists.txt
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(OpenCAL-ALL)


option(BUILD_OPENCAL_SERIAL "Build the OpenCAL serial version"                                      ON )
option(BUILD_OPENCAL_OMP    "Build the OpenCAL-OMP OpenMP parallel version (OpenMP required)"       ON )
option(BUILD_OPENCAL_CL     "Build the OpenCAL-CL OpenCL parallel version (OpenCL required)"        ON )
option(BUILD_OPENCAL_GL     "Build the OpenCAL-GL visualization library (OpenGL and GLUT required)" ON )
option(BUILD_EXAMPLES       "Build the examples for each OpenCAL version"                           ON )
option(BUILD_DOCUMENTATION  "Build the HTML based API documentation (requires Doxygen)"             OFF)

if (WIN32)
	OPTION(USE_EXTERN_FREEGLUT_STATIC "Link the freeglut Win32 lib, included in the extern directory, statically (set to ON only if you use MS Visual Studio)" ON)
	OPTION(USE_EXTERN_PTW32_STATIC    "Link the pthread Win32 lib, included in the extern directory, statically (set to ON only if you use MS Visual Studio)"  ON)	
endif()


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

if(BUILD_DOCUMENTATION)
	find_package(Doxygen)
endif(BUILD_DOCUMENTATION)


cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

find_package(OpenCAL 1.1 COMPONENTS  OpenCAL OpenCAL-CL OpenCAL-GL REQUIRED)
find_package(OpenCL  REQUIRED)
find_package(OpenGL  REQUIRED)
find_package(GLUT REQUIRED)
find_package(Threads REQUIRED)

if(NOT MSVC)
  find_library(M_LIB m)
else()
  # not needed on windows
  set(M_LIB "")
endif()

FIND_PACKAGE(OpenMP)
if(OPENMP_FOUND)
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}/include
  ${OPENCAL_INCLUDE_DIR}
	${OPENGL_INCLUDE_DIRS}
	${GLUT_INCLUDE_DIR}
)
include_directories( ${OpenCL_INCLUDE_DIRS} )

file(GLOB SRC
    "include/*.h"
    "source/*.cpp"
    "source/*.c"
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

add_executable(calcl-sciddicaS3hex ${SRC})

target_link_libraries(calcl-sciddicaS3hex ${OPENCAL_LIBRARIES} ${M_LIB} ${OPENGL_LIBRARIES} ${GLUT_glut_LIBRARY} ${OpenCL_LIBRARY} pthread)

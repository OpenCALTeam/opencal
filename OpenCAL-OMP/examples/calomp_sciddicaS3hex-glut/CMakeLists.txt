# opencal/OpenCAL/examples/calomp_sciddicaS3hex-glut/CMakeLists.txt
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_library(M_LIB m)

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/source)
include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}/include
	${OPENGL_INCLUDE_DIRS}
	${GLUT_INCLUDE_DIR}
)

LINK_DIRECTORIES(${PROJECT_SOURCE_DIR}/lib)

file(GLOB SRC
    "include/*.h"
    "source/*.c"
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

add_executable(calomp_sciddicaS3hex-glut ${SRC})
target_link_libraries(calomp_sciddicaS3hex-glut  opencal-omp ${M_LIB} ${OPENGL_LIBRARIES} ${GLUT_glut_LIBRARY} )

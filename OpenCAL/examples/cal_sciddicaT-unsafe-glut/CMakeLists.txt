# opencal/OpenCAL/examples/cal_sciddicaT-unsafe-glut/CMakeLists.txt
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

add_executable(cal_sciddicaT-unsafe-glut ${SRC})
target_link_libraries(cal_sciddicaT-unsafe-glut opencal ${M_LIB} ${OPENGL_LIBRARIES} ${GLUT_glut_LIBRARY} )

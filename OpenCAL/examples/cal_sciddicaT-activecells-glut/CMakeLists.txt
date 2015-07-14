# opencal/OpenCAL/examples/cal_sciddicaT-activecells-glut/CMakeLists.txt
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

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


add_executable(cal_sciddicaT-activecells-glut ${SRC})
target_link_libraries(cal_sciddicaT-activecells-glut opencal ${OPENGL_LIBRARIES} ${GLUT_glut_LIBRARY} )

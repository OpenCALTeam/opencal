# opencal/OpenCAL/examples/cal_iso-3src-glut/CMakeLists.txt
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

link_directories(
	${PROJECT_SOURCE_DIR}/lib
	${CMAKE_CURRENT_SOURCE_DIR}/source
)

include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}/include
	${OPENGL_INCLUDE_DIRS}
	${GLUT_INCLUDE_DIR}
)

file(GLOB SRC
    "include/*.h"
    "source/*.c"
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

add_executable(calomp_belousov-zhabotisnky-reaction-glut ${SRC})
target_link_libraries(calomp_belousov-zhabotisnky-reaction-glut opencal-omp ${OPENGL_LIBRARIES} ${GLUT_glut_LIBRARY} )

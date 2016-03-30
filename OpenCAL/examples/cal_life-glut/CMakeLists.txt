# opencal/OpenCAL/examples/cal_iso-3src-glut/CMakeLists.txt
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}")
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(OpenCAL 1.0 COMPONENTS  OpenCAL  REQUIRED)

include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}/include
	${OPENGL_INCLUDE_DIRS}
	${GLUT_INCLUDE_DIR}
	${OPENCAL_INCLUDE_DIR}
)

message(status sdljvhskdljhbv	${OPENCAL_INCLUDE_DIR} ${OPENCAL_LIBRARIES})

file(GLOB SRC
    "include/*.h"
    "source/*.c"
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

add_executable(cal_life-glut ${SRC})
target_link_libraries(cal_life-glut ${OPENCAL_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_glut_LIBRARY} )

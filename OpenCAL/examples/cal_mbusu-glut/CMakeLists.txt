# opencal/OpenCAL/examples/cal_mbusu-glut/CMakeLists.txt
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

find_package(OpenGL  REQUIRED)
find_package(GLUT REQUIRED)
find_library(M_LIB m)
find_package(OpenCAL 1.0 COMPONENTS  OpenCAL  REQUIRED)


include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}/include
	${OPENGL_INCLUDE_DIRS}
	${GLUT_INCLUDE_DIR}
	${OPENCAL_INCLUDE_DIR}

)

file(GLOB SRC
    "include/*.h"
    "source/*.c"
)

add_executable(cal_mbusu-glut ${SRC})
target_link_libraries(cal_mbusu-glut 	${OPENCAL_LIBRARIES}
 																			${M_LIB}
																			${OPENGL_LIBRARIES}
																	 		${GLUT_glut_LIBRARY}
																 			${CMAKE_THREAD_LIBS_INIT})

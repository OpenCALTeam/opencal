# opencal/OpenCAL/examples/cal_mod2CA3D-glut/CMakeLists.txt
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

#resolve the required dependencies
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(OpenCAL 1.0 COMPONENTS  OpenCAL  REQUIRED)


include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}/include
	${OPENGL_INCLUDE_DIRS}
	${GLUT_INCLUDE_DIR}
	${OPENCAL_INCLUDE_DIR}
)

LINK_DIRECTORIES(${PROJECT_SOURCE_DIR}/lib)

file(GLOB SRC
    "include/*.h"
    "source/*.c"
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

add_executable(cal_mod2CA3D-glut ${SRC})
target_link_libraries(cal_mod2CA3D-glut ${OPENCAL_LIBRARIES}
 																				${OPENGL_LIBRARIES}
																			 	${GLUT_glut_LIBRARY} )

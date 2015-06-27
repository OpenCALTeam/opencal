# opencal/OpenCAL/examples/mod2CA3D-glut/CMakeLists.txt
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

find_package(OpenGL REQUIRED)
if (${USE_EXTERN_FREEGLUT_STATIC} STREQUAL "ON")
	set (GLUT_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/../extern/freeglut-2.8.1/include")
	set (GLUT_glut_LIBRARY "${PROJECT_SOURCE_DIR}/../extern/freeglut-2.8.1/lib/x86/freeglut_static.lib")
	add_definitions(-DFREEGLUT_STATIC)
else()
	find_package(GLUT REQUIRED)
endif()

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/source)
include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}/include
	${OPENGL_INCLUDE_DIRS}
	${GLUT_INCLUDE_DIR}
)

LINK_DIRECTORIES(${PROJECT_SOURCE_DIR}/lib)

file(GLOB SRC
    "include/*.h"
    "source/*.cpp"
    "source/*.c"
)


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

add_executable(mod2CA3D-glut ${SRC})
target_link_libraries(mod2CA3D-glut opencal ${OPENGL_LIBRARIES} ${GLUT_glut_LIBRARY} )



# opencal/OpenCAL/examples/sciddicaT-activecells-glut/CMakeLists.txt

cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

if (${FREEGLUT_STATIC} STREQUAL "ON")
	add_definitions(-DFREEGLUT_STATIC)
endif()

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/source)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

LINK_DIRECTORIES(${PROJECT_SOURCE_DIR}/lib)

file(GLOB SRC
    "include/*.h"
    "source/*.cpp"
    "source/*.c"
)

add_executable(sciddicaT-activecells-glut ${SRC})

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
include_directories( ${OPENGL_INCLUDE_DIRS}  ${GLUT_INCLUDE_DIR} )


target_link_libraries(sciddicaT-activecells-glut opencal ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} )



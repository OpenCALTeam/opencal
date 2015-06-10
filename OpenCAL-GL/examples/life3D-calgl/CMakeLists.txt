cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package (Threads REQUIRED)
find_library(M_LIB m)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/source)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)


LINK_DIRECTORIES(${PROJECT_SOURCE_DIR}/OpenCAL/lib)
LINK_DIRECTORIES(${PROJECT_SOURCE_DIR}/OpenCAL-GL/lib)

include_directories(${PROJECT_SOURCE_DIR}/OpenCAL/include)
include_directories(${PROJECT_SOURCE_DIR}/OpenCAL-GL/include)

include_directories( ${OPENGL_INCLUDE_DIRS}  ${GLUT_INCLUDE_DIRS} )

file(GLOB SRC
    "include/*.h"
    "source/*.cpp"
    "source/*.c"
)

add_executable(life3D-calgl ${SRC})
target_link_libraries(life3D-calgl ${M_LIB}  libopencal-gl.a libopencal.a ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} ${CMAKE_THREAD_LIBS_INIT})



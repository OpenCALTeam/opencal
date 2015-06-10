cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package (Threads REQUIRED)

FIND_PACKAGE(OpenMP)
IF(OPENMP_FOUND)
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
ENDIF()

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

add_executable(sciddicaT-calgl ${SRC})
target_link_libraries(sciddicaT-calgl -lm   libopencal-gl.a libopencal.a ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} ${CMAKE_THREAD_LIBS_INIT})



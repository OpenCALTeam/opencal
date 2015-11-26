# opencal/OpenCAL-GL/examples/calgl_sciddicaT_explicit/CMakeLists.txt
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)


find_package(OpenGL  REQUIRED)
find_package(GLUT REQUIRED)
find_package(Threads REQUIRED)
find_library(M_LIB m)

FIND_PACKAGE(OpenMP)
if(OPENMP_FOUND)
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}/include
	${PROJECT_SOURCE_DIR}/OpenCAL/include
	${PROJECT_SOURCE_DIR}/OpenCAL-GL/include
	${OPENGL_INCLUDE_DIRS}
	${GLUT_INCLUDE_DIR}
)

link_directories(
	${CMAKE_CURRENT_SOURCE_DIR}/source
	${PROJECT_SOURCE_DIR}/OpenCAL/lib
	${PROJECT_SOURCE_DIR}/OpenCAL-GL/lib
)

file(GLOB SRC
    "include/*.h"
    "source/*.cpp"
    "source/*.c"
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

add_executable(calgl_sciddicaT_explicit ${SRC})

target_link_libraries(calgl_sciddicaT_explicit opencal-gl opencal ${M_LIB} ${OPENGL_LIBRARIES} ${GLUT_glut_LIBRARY} ${CMAKE_THREAD_LIBS_INIT})

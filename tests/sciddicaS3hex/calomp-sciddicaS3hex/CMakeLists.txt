cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

find_package(OpenCAL 1.1 COMPONENTS  OpenCAL-OMP REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

if(NOT MSVC)
  find_library(M_LIB m)
else()
  # not needed on windows
  set(M_LIB "")
endif()

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/source)
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

IF(BUILD_OPENCAL_OMP_PARALLEL)
	FIND_PACKAGE(OpenMP REQUIRED)
	IF(OPENMP_FOUND)
		SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
		SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
		set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
	ENDIF()
ENDIF()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

add_executable(calomp-sciddicaS3hex ${SRC})
target_link_libraries(calomp-sciddicaS3hex  ${OPENCAL_LIBRARIES} ${M_LIB} ${OPENGL_LIBRARIES} ${GLUT_glut_LIBRARY} )

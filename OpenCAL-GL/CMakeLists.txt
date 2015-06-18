# opencal/OpenCAL-GL/CMakeLists.txt

cmake_minimum_required(VERSION 2.8 FATAL_ERROR)


find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
include_directories( ${OPENGL_INCLUDE_DIRS} ${GLUT_INCLUDE_DIR} )
include_directories( ${PROJECT_SOURCE_DIR}/OpenCAL/include )

if (${PTW32_STATIC} STREQUAL "ON")
	add_definitions(-DPTW32_STATIC_LIB)
	include_directories( ${PROJECT_SOURCE_DIR}/extern/pthread-win32/include )
endif()

if (${FREEGLUT_STATIC} STREQUAL "ON")
	add_definitions(-DFREEGLUT_STATIC)
endif()


link_directories(${CMAKE_CURRENT_SOURCE_DIR}/source)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

file(GLOB SRC
    "include/*.h"
    "source/*.cpp"
    "source/*.c"
)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

add_library(opencal-gl STATIC ${SRC} )


if(${EXAMPLES} STREQUAL "ON")
        add_subdirectory(examples)
else()
      	MESSAGE(STATUS "${CMAKE_CURRENT_SOURCE_DIR}/examples EXCLUDED FROM COMPILATION")
endif()


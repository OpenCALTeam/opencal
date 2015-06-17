cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(OpenCAL)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

if (WIN32)
	add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif(WIN32)

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/source)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

file(GLOB SRC
    "include/*.h"
    "source/*.cpp"
    "source/*.c"
)

add_library(opencal STATIC ${SRC})

if(${EXAMPLES} STREQUAL "ON")
	add_subdirectory(examples)
else()
	MESSAGE(STATUS "${CMAKE_CURRENT_SOURCE_DIR}/examples EXCLUDED FROM COMPILATION")	
endif()

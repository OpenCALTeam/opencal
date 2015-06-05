cmake_minimum_required(VERSION 2.8 FATAL_ERROR)



include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/../OpenCAL/include )



set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)


link_directories(${CMAKE_CURRENT_SOURCE_DIR}/source)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

file(GLOB SRC
    "include/*.h"
    "source/*.cpp"
    "source/*.c"
)

add_library(opencal-gl STATIC ${SRC})

add_subdirectory(examples)

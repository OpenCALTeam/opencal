cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(OpenCAL)


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

add_library(opencal STATIC ${SRC})
target_link_libraries(opencal omp)

if(${EXAMPLES} STREQUAL "YES")
        add_subdirectory(examples)
else()
	MESSAGE(STATUS "${CMAKE_CURRENT_SOURCE_DIR}/examples EXCLUDED FROM COMPILATION")	

endif()

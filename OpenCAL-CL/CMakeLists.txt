cmake_minimum_required(VERSION 2.8 FATAL_ERROR)


find_package( OpenCL REQUIRED )

#first compile opencl-utils
add_subdirectory(OpenCL-Utils)

#include_directories( "$ENV{CUDA_PATH}/include" )
#MESSAGE( STATUS "opencl includes: " "$ENV{CUDA_PATH}/include" )
include_directories( ${OpenCL_INCLUDE_DIRS} )
include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/../OpenCAL/include )
include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/OpenCL-Utils/include)


MESSAGE( STATUS "opencl includes: " ${OpenCL_INCLUDE_DIRS} )
MESSAGE( STATUS "opencl includes: " ${OpenCL_LIBRARIES} )



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


add_library(opencal-cl STATIC ${SRC})


if(${EXAMPLES} STREQUAL "YES")
        add_subdirectory(examples)
else()
      	MESSAGE(STATUS "${CMAKE_CURRENT_SOURCE_DIR}/examples EXCLUDED FROM COMPILATION")

endif()


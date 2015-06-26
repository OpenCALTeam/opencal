# opencal/OpenCAL-CL/OpenCL-Utils/CMakeLists.txt
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)


find_package( OpenCL REQUIRED )


include_directories( ${OpenCL_INCLUDE_DIRS} )

if(WIN32)
	include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/dirent4vs )
endif()


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


add_library(opencl_utils STATIC ${SRC})

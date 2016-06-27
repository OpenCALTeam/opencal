#
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
find_package(OpenCAL 1.0 COMPONENTS OpenCAL  OpenCAL-CL REQUIRED)
find_package(OpenCL REQUIRED)

if(NOT MSVC)
  find_library(M_LIB m)
else()
  # not needed on windows
  set(M_LIB "")
endif()

include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
	${OPENCAL_INCLUDE_DIR}
)
include_directories( ${OpenCL_INCLUDE_DIRS} )

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/source)
link_directories(${OpenCL_LIBRARIES})

file(GLOB SRC
    "include/*.h"
    "source/*.c"
)

add_executable(calcl-sciddicaT-activecells-test ${SRC})
target_link_libraries(calcl-sciddicaT-activecells-test ${M_LIB} ${OpenCL_LIBRARY} ${OPENCAL_LIBRARIES} )

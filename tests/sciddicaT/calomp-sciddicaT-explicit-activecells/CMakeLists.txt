cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

find_package(OpenCAL 1.0 COMPONENTS  OpenCAL-OMP  REQUIRED)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/source)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include
                    ${OPENCAL_INCLUDE_DIR}
                    ${CMAKE_CURRENT_SOURCE_DIR}/../../include)


file(GLOB SRC
    "include/*.h"
    "source/*.c"
)

add_executable(calomp-sciddicaT-explicit-activecells-test ${SRC})
target_link_libraries(calomp-sciddicaT-explicit-activecells-test ${OPENCAL_LIBRARIES})

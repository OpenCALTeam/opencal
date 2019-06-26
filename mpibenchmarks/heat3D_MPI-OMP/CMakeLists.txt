cmake_minimum_required(VERSION 2.8)

project(calompmpi_heat)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/../../cmake")

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)

if(COMPILER_SUPPORTS_CXX14)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
else()
        message(STATUS "OPENCAL++ - The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

# Require MPI for this project:
find_package(MPI REQUIRED)
find_package(OpenCAL 1.1 COMPONENTS  OpenCAL-OMP REQUIRED)


if(NOT MSVC)
  find_library(M_LIB m)
else()
  # not needed on windows
  set(M_LIB "")
endif()
include_directories( 
                     ${OPENCAL_INCLUDE_DIR}
                   )
include_directories(${MPI_INCLUDE_PATH})

set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})

set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS})

include_directories(${MPI_INCLUDE_PATH})

add_executable(${PROJECT_NAME}  heat3D.cpp)
target_link_libraries(${PROJECT_NAME} ${MPI_LIBRARIES} ${OPENCAL_LIBRARIES} ${M_LIB})

cmake_minimum_required(VERSION 2.8)

project(calclmpi_modCA3D)

message(STATUS "cmake_module_path =  ${CMAKE_MODULE_PATH}" )
message(STATUS "cmake_prefix_path =  ${CMAKE_PREFIX_PATH}" )
message(STATUS "CMAKE_FRAMEWORK_PATH =  ${CMAKE_FRAMEWORK_PATH}" )
message(STATUS "CMAKE_APPBUNDLE_PATH =  ${CMAKE_APPBUNDLE_PATH}" )
message(STATUS "CMAKE_SYSTEM_PREFIX_PATH =  ${CMAKE_SYSTEM_PREFIX_PATH}" )

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)

if(COMPILER_SUPPORTS_CXX14)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
else()
        message(STATUS "OPENCAL++ - The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

FIND_PACKAGE(OpenMP REQUIRED)
	IF(OPENMP_FOUND)
		SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
		SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
		set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
	ENDIF()

# Require MPI for this project:
find_package(MPI REQUIRED)
find_package(OpenCAL 1.1 COMPONENTS  OpenCAL OpenCAL-CL REQUIRED)

find_package( OpenCL REQUIRED )

if(NOT MSVC)
  find_library(M_LIB m)
else()
  # not needed on windows
  set(M_LIB "")
endif()
include_directories( ${OpenCL_INCLUDE_DIRS}
                     ${OPENCAL_INCLUDE_DIR}
                   )
include_directories(${MPI_INCLUDE_PATH})

set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})

set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS})

include_directories(MPI_INCLUDE_PATH)

add_executable(${PROJECT_NAME} mod2CA3DMPI.cpp)
target_link_libraries(${PROJECT_NAME} ${MPI_LIBRARIES} ${OPENCAL_LIBRARIES} ${M_LIB} ${OpenCL_LIBRARIES})

# opencal/OpenCAL/examples/cal_life/CMakeLists.txt
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/source)
include_directories(${PROJECT_SOURCE_DIR}/OpenCAL-OMP/include)

FIND_PACKAGE(OpenMP)
IF(OPENMP_FOUND)
	SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
ENDIF()

link_directories(${PROJECT_SOURCE_DIR}/OpenCAL-OMP/lib)
file(GLOB SRC
    "source/*.cpp"
    "OpenCAL-OMP/include/*.h"
)
add_executable(calomp-identity3D-test ${SRC})
target_link_libraries(calomp-identity3D-test opencal-omp)

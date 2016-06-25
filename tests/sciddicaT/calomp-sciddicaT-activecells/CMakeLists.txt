cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

find_package(OpenCAL 1.0 COMPONENTS  OpenCAL-OMP REQUIRED)

IF(BUILD_OPENCAL_OMP_PARALLEL)
	FIND_PACKAGE(OpenMP REQUIRED)
	IF(OPENMP_FOUND)
		SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
		SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
		set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
	ENDIF()
ENDIF()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/source)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${OPENCAL_INCLUDE_DIR}
                    ${CMAKE_CURRENT_SOURCE_DIR}/../../include)

file(GLOB SRC
    "source/*.c"
)

add_executable(calomp-sciddicaT-activecells-test ${SRC})
target_link_libraries(calomp-sciddicaT-activecells-test ${OPENCAL_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

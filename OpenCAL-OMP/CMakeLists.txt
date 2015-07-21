# opencal/OpenCAL-OMP/CMakeLists.txt
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

# add a target to generate API documentation with Doxygen
if(BUILD_DOCUMENTATION)
    if(NOT DOXYGEN_FOUND)
        message(FATAL_ERROR "Doxygen is needed to build the documentation.")
    endif()

    set(doxyfile_in ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
    set(doxyfile ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

    configure_file(${doxyfile_in} ${doxyfile} @ONLY)

    add_custom_target(opencal-omp-doc ALL
        COMMAND ${DOXYGEN_EXECUTABLE} ${doxyfile}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen for for OpenCAL-OMP"
        VERBATIM)

    	install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html DESTINATION share/doc/OpenCAL/OpenCAL-OMP)
endif()


FIND_PACKAGE(OpenMP REQUIRED)
IF(OPENMP_FOUND)
	SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
ENDIF()


set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/source)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

file(GLOB SRC
    "include/*.h"
     "include/OpenCAL-OMP/*.h"    
    "source/*.c"
)
file(GLOB HEADERS
    "include/*.h"
    "include/OpenCAL-OMP/*.h"
)

add_library(opencal-omp ${SHAREDORSTATIC} ${SRC})
install(TARGETS opencal-omp DESTINATION ${INSTALL_LIB_DIR})
install(FILES ${HEADERS} DESTINATION ${INSTALL_INCLUDE_DIR}/OpenCAL-OMP)

if(${BUILD_EXAMPLES} STREQUAL "ON")
	add_subdirectory(examples)
endif()


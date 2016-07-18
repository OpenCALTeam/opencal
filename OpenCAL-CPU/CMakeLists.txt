# opencal/OpenCAL/CMakeLists.txt
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

set(OPENCAL-CPU_LIBRARY_NAME OpenCAL-CPU CACHE PATH "Name of the specific version of opencal (CPU)")

# add a target to generate API documentation with Doxygen
if(BUILD_DOCUMENTATION)
    if(NOT DOXYGEN_FOUND)
        message(FATAL_ERROR "Doxygen is needed to build the documentation.")
    endif()

    set(doxyfile_in ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
    set(doxyfile ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

    configure_file(${doxyfile_in} ${doxyfile} @ONLY)

    add_custom_target(opencal-doc ALL
        COMMAND ${DOXYGEN_EXECUTABLE} ${doxyfile}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen for for OpenCAL"
        VERBATIM)
        install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html DESTINATION ${INSTALL_DOCS_DIR}/${OPENCAL_LIBRARY_NAME})
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
    "include/OpenCAL-CPU/*.h"
    "source/*.c"
)

file(GLOB HEADERS
    "include/OpenCAL-CPU/*.h"
)

add_library(opencal-cpu ${SHAREDORSTATIC} ${SRC})
install(TARGETS opencal-cpu DESTINATION  ${INSTALL_LIB_DIR})
install(FILES ${HEADERS} DESTINATION ${INSTALL_INCLUDE_DIR}/${OPENCAL-CPU_LIBRARY_NAME})

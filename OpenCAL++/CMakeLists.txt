# opencal/OpenCAL++14/CMakeLists.txt
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

set(OPENCAL++_LIBRARY_NAME OpenCAL++ CACHE PATH "Name of the specific version of opencal (OPENCAL GL)")

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)
if(COMPILER_SUPPORTS_CXX14)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
else()
        message(STATUS "OPENCAL++ - The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

# add a target to generate API documentation with Doxygen
if(BUILD_DOCUMENTATION)
    if(NOT DOXYGEN_FOUND)
        message(FATAL_ERROR "Doxygen is needed to build the documentation.")
    endif()

    set(doxyfile_in ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
    set(doxyfile ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

    configure_file(${doxyfile_in} ${doxyfile} @ONLY)

    add_custom_target(opencalpp14-doc ALL
        COMMAND ${DOXYGEN_EXECUTABLE} ${doxyfile}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen for for OpenCAL++"
        VERBATIM)

    	install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html DESTINATION share/doc)
endif()


set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

if (WIN32)
	add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif(WIN32)

set( OPENCAL_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/include )

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/source)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

file(GLOB SRC
    "include/*.h"
    "source/*.cpp"
    "source/*.c"
    "include/OpenCAL++/*.h"


)

file(GLOB HEADERS
    "include/*.h"
    "include/OpenCAL++/*.h"
)
add_library(opencal++ ${SHAREDORSTATIC} ${SRC})
install(TARGETS opencal++ DESTINATION ${INSTALL_LIB_DIR})
install(FILES ${HEADERS} DESTINATION ${INSTALL_INCLUDE_DIR}/${OPENCAL++_LIBRARY_NAME})

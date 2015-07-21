# opencal/OpenCAL-CL/CMakeLists.txt
cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

# add a target to generate API documentation with Doxygen
if(BUILD_DOCUMENTATION)
    if(NOT DOXYGEN_FOUND)
        message(FATAL_ERROR "Doxygen is needed to build the documentation.")
    endif()

    set(doxyfile_in ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
    set(doxyfile ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

    configure_file(${doxyfile_in} ${doxyfile} @ONLY)

    add_custom_target(opencal-cl-doc ALL
        COMMAND ${DOXYGEN_EXECUTABLE} ${doxyfile}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen for for OpenCAL-CL"
        VERBATIM)

    	install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html DESTINATION share/doc/OpenCAL/OpenCAL-CL)
endif()


find_package( OpenCL REQUIRED )

#compile opencl-utils first
add_subdirectory(OpenCL-Utils)

set (OPENCL-UTILS_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/OpenCL-Utils/include")
set (OPENCAL_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/OpenCAL/include")

include_directories(
	${OpenCL_INCLUDE_DIRS}
	${OPENCL-UTILS_INCLUDE_DIR}
	${OPENCAL_INCLUDE_DIR}
)

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

file(GLOB HEADERS
    "include/*.h"
    "include/OpenCAL-CL/*.h"
)

file(GLOB KERNELS_HEADERS
    "kernel/include/*.h"
)

file(GLOB KERNELS_SRC
    "kernel/source/*.cl"
    "kernel/source/*.c"
)

add_library(opencal-cl ${SHAREDORSTATIC} ${SRC})
install(TARGETS opencal-cl DESTINATION ${INSTALL_LIB_DIR})
install(FILES ${HEADERS} DESTINATION ${INSTALL_INCLUDE_DIR}/OpenCAL-CL)

install(FILES ${KERNELS_HEADERS} DESTINATION ${INSTALL_SHARE_DIR}/OpenCAL-CL/kernel/include)
install(FILES ${KERNELS_SRC} DESTINATION ${INSTALL_SHARE_DIR}/OpenCAL-CL/kernel/source)


if(${BUILD_EXAMPLES} STREQUAL "ON")
        add_subdirectory(examples)
		install(DIRECTORY examples DESTINATION ${INSTALL_EXAMPLE_DIR}/OpenCAL-CL)
endif()

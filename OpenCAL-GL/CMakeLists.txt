# opencal/OpenCAL-GL/CMakeLists.txt
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)


find_package(OpenGL REQUIRED)

if(${USE_EXTERN_FREEGLUT_STATIC} STREQUAL "ON")
	set (GLUT_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/extern/freeglut-2.8.1/include")
	set (GLUT_glut_LIBRARY "${PROJECT_SOURCE_DIR}/extern/freeglut-2.8.1/lib/x86/freeglut_static.lib")
	add_definitions(-DFREEGLUT_STATIC)
else()
	find_package(GLUT REQUIRED)
endif()

if (${USE_EXTERN_PTW32_STATIC} STREQUAL "ON")
	set (PTW32_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/extern/pthread-win32/include")
	set (PTW32_ptw32_LIBRARY "${PROJECT_SOURCE_DIR}/extern/pthread-win32/lib/x86/pthread_lib.lib")
	add_definitions(-DPTW32_STATIC_LIB)
else()
	find_package(Threads REQUIRED)
endif()

set (OPENCAL_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/OpenCAL/include")
set (OPENCAL-GL_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

include_directories(
	${OPENCAL-GL_INCLUDE_DIR}
	${OPENCAL_INCLUDE_DIR}
	${OPENGL_INCLUDE_DIRS}
	${GLUT_INCLUDE_DIR}
	${PTW32_INCLUDE_DIR}
)

link_directories(
	${CMAKE_CURRENT_SOURCE_DIR}/source
)

file(GLOB SRC
    "include/*.h"
    "source/*.cpp"
    "source/*.c"
)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

if (WIN32)
	add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif(WIN32)

add_library(opencal-gl STATIC ${SRC} )


if(${BUILD_EXAMPLES} STREQUAL "ON")
        add_subdirectory(examples)
else()
      	MESSAGE(STATUS "${CMAKE_CURRENT_SOURCE_DIR}/examples EXCLUDED FROM COMPILATION")
endif()

# add a target to generate API documentation with Doxygen
if(BUILD_DOCUMENTATION)
    if(NOT DOXYGEN_FOUND)
        message(FATAL_ERROR "Doxygen is needed to build the documentation.")
    endif()

    set(doxyfile_in ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
    set(doxyfile ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

    configure_file(${doxyfile_in} ${doxyfile} @ONLY)

    add_custom_target(opencal-gl-doc ALL
        COMMAND ${DOXYGEN_EXECUTABLE} ${doxyfile}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen for OPENCAL-GL"
        VERBATIM)

    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html DESTINATION share/doc)
endif()


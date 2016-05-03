# opencal/OpenCAL-GL/CMakeLists.txt
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

set(OPENCAL-GL_LIBRARY_NAME OpenCAL-GL CACHE PATH "Name of the specific version of opencal (OPENCAL GL)")

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

    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html DESTINATION ${INSTALL_DOCS_DIR}/${OPENCAL-GL_LIBRARY_NAME})
endif()


find_package(Threads REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

set (OPENCAL_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/OpenCAL/include")
set (OPENCAL-GL_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/source)
include_directories(
	${OPENCAL-GL_INCLUDE_DIR}
	${OPENCAL_INCLUDE_DIR}
	${OPENGL_INCLUDE_DIRS}
	${GLUT_INCLUDE_DIR}
)
file(GLOB SRC "source/*.c")
file(GLOB HEADERS "include/OpenCAL-GL/*.h")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY bin)

if ( ENABLE_STATIC )
  add_library(opencal-gl_static STATIC ${SRC} ${HEADERS})
  install(TARGETS opencal-gl_static DESTINATION  ${INSTALL_LIB_DIR})
endif()
if ( ENABLE_SHARED )
  add_library(opencal-gl SHARED ${SRC} ${HEADERS})
  target_link_libraries(opencal-gl pthread opencal)
  if (MSVC)
    install(TARGETS opencal-gl
            ARCHIVE DESTINATION  ${INSTALL_LIB_DIR}
            RUNTIME DESTINATION  ${INSTALL_BIN_DIR}
            )
  else()
    install(TARGETS opencal-gl DESTINATION  ${INSTALL_LIB_DIR})
  endif()
endif()
install(FILES ${HEADERS} DESTINATION ${INSTALL_INCLUDE_DIR}/${OPENCAL-GL_LIBRARY_NAME})

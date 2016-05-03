# opencal/OpenCAL/CMakeLists.txt
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

set(OPENCAL_LIBRARY_NAME OpenCAL CACHE PATH "Name of the specific version of opencal (SERIAL)")

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

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY bin)

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/source)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
file(GLOB SRC "source/*.c")
file(GLOB HEADERS "include/OpenCAL/*.h")

if ( ENABLE_STATIC )
  add_library(opencal_static STATIC ${SRC} ${HEADERS})
  install(TARGETS opencal_static DESTINATION  ${INSTALL_LIB_DIR})
endif()
if ( ENABLE_SHARED )
  add_library(opencal SHARED ${SRC} ${HEADERS})
  if (MSVC)
    install(TARGETS opencal
            ARCHIVE DESTINATION  ${INSTALL_LIB_DIR}
            RUNTIME DESTINATION  ${INSTALL_BIN_DIR}
          )
  else()
    install(TARGETS opencal DESTINATION  ${INSTALL_LIB_DIR})
  endif()
endif()

install(FILES ${HEADERS} DESTINATION ${INSTALL_INCLUDE_DIR}/${OPENCAL_LIBRARY_NAME})

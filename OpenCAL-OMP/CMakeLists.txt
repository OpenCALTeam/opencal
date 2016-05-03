# opencal/OpenCAL-OMP/CMakeLists.txt
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

set(OPENCAL-OMP_LIBRARY_NAME OpenCAL-OMP CACHE PATH "Name of the specific version of opencal (OMP)")

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
    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html DESTINATION ${INSTALL_DOCS_DIR}/${OPENCAL-OMP_LIBRARY_NAME})
endif()


IF(BUILD_OPENCAL_OMP_PARALLEL)
	FIND_PACKAGE(OpenMP REQUIRED)
	IF(OPENMP_FOUND)
		SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
		SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
		set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
	ENDIF()
ENDIF()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY bin)

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/source)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
file(GLOB SRC "source/*.c")
file(GLOB HEADERS "include/OpenCAL-OMP/*.h")

if ( ENABLE_STATIC )
  add_library(opencal-omp_static STATIC ${SRC} ${HEADERS})
  install(TARGETS opencal-omp_static DESTINATION  ${INSTALL_LIB_DIR})
endif()
if ( ENABLE_SHARED )
  add_library(opencal-omp SHARED ${SRC} ${HEADERS})
  if (MSVC)
    install(TARGETS opencal-omp
            ARCHIVE DESTINATION  ${INSTALL_LIB_DIR}
            RUNTIME DESTINATION  ${INSTALL_BIN_DIR}
            )
  else()
    install(TARGETS opencal-omp DESTINATION  ${INSTALL_LIB_DIR})
  endif()
endif()
install(FILES ${HEADERS} DESTINATION ${INSTALL_INCLUDE_DIR}/${OPENCAL-OMP_LIBRARY_NAME})

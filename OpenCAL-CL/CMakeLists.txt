# opencal/OpenCAL-CL/CMakeLists.txt
cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

set(OPENCAL-CL_LIBRARY_NAME OpenCAL-CL CACHE PATH "Name of the specific version of opencal (OPENCAL CL)")

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

    add_custom_target(opencal-cl-doc ALL
        COMMAND ${DOXYGEN_EXECUTABLE} ${doxyfile}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen for for OpenCAL-CL"
        VERBATIM)

    	install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html DESTINATION ${INSTALL_DOCS_DIR}/${OPENCAL-CL_LIBRARY_NAME})
endif()

if(NOT MSVC)
  find_library(M_LIB m)
else()
  # not needed on windows
  set(M_LIB "")
endif()


find_package(OpenCL REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

# Require MPI for this project:
find_package(MPI REQUIRED)
set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})
set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS})
include_directories(MPI_INCLUDE_PATH)




set (OPENCAL_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/OpenCAL/include")

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/source)
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
	${OPENCAL_INCLUDE_DIR}
  ${OpenCL_INCLUDE_DIRS}
	${OPENGL_INCLUDE_DIRS}
	${GLUT_INCLUDE_DIR}
)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY bin)


file(GLOB SRC "source/*.c")
file(GLOB HEADERS "include/OpenCAL-CL/*.h")
file(GLOB KERNELS_HEADERS "kernel/include/OpenCAL-CL/*.h")
file(GLOB KERNELS_SRC
  "kernel/source/*.cl"
  "kernel/source/*.c*"
)

if ( ENABLE_STATIC )
  add_library(opencal-cl_static STATIC ${SRC} ${HEADERS})
  install(TARGETS opencal-cl_static DESTINATION  ${INSTALL_LIB_DIR})
endif()
if ( ENABLE_SHARED )
  add_library(opencal-cl SHARED ${SRC} ${HEADERS})
  target_link_libraries(opencal-cl pthread opencal ${OpenCL_LIBRARY} ${M_LIB})
  if (MSVC)
    install(TARGETS opencal-cl
            ARCHIVE DESTINATION  ${INSTALL_LIB_DIR}
            RUNTIME DESTINATION  ${INSTALL_BIN_DIR}
            )
  else()
    install(TARGETS opencal-cl DESTINATION  ${INSTALL_LIB_DIR})
  endif()
endif()

install(FILES ${HEADERS} DESTINATION ${INSTALL_INCLUDE_DIR}/${OPENCAL-CL_LIBRARY_NAME})

install(FILES ${KERNELS_HEADERS} DESTINATION ${INSTALL_INCLUDE_DIR}/${OPENCAL-CL_LIBRARY_NAME}/kernel/include/${OPENCAL-CL_LIBRARY_NAME})
install(FILES ${KERNELS_SRC} DESTINATION ${INSTALL_INCLUDE_DIR}/${OPENCAL-CL_LIBRARY_NAME}/kernel/source)

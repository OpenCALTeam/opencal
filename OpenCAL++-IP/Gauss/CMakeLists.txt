 cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

# CGAL and its components
find_package( CGAL QUIET COMPONENTS  )

if ( NOT CGAL_FOUND )

  message(STATUS "This project requires the CGAL library, and will not be compiled.")
  return()  

endif()

# include helper file
include( ${CGAL_USE_FILE} )


# Boost and its components
find_package( Boost REQUIRED )

if ( NOT Boost_FOUND )

  message(STATUS "This project requires the Boost library, and will not be compiled.")

  return()  

endif() 
 
 
 
 
find_package(OpenCAL 1.0 COMPONENTS OpenCAL++ REQUIRED)
find_package( OpenCV REQUIRED )
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)

if(COMPILER_SUPPORTS_CXX14)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
else()
        message(STATUS "OPENCAL++ - The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()


#project( opencal++_CAUSS )

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/source)
#include_directories(${PROJECT_SOURCE_DIR}/OpenCAL++-IP/include/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${OPENCAL_INCLUDE_DIR} )
#include_directories(${GTEST_INCLUDE_DIRS})
#include_directories(${CMAKE_CURRENT_SOURCE_DIR}/)



file(GLOB SRC
    "include/*.h"
    "source/*.cpp"
    "source/*.h"
)



message(STATUS  "sdfvsdfsdf{PROJECT_SOURCE_DIR}/OpenCAL++/include/" )
add_executable(gauss ${SRC} )
#set_property(TARGET opencal++_imageProcessing PROPERTY CXX_STANDARD 14)
target_link_libraries(gauss ${OpenCV_LIBS})


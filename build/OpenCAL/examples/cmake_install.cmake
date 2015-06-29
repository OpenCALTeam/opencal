# Install script for directory: C:/Users/Donato/Documents/git/opencal/OpenCAL/examples

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "C:/Program Files (x86)/OpenCAL-ALL")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("C:/Users/Donato/Documents/git/opencal/build/OpenCAL/examples/iso-3src-glut/cmake_install.cmake")
  include("C:/Users/Donato/Documents/git/opencal/build/OpenCAL/examples/life/cmake_install.cmake")
  include("C:/Users/Donato/Documents/git/opencal/build/OpenCAL/examples/life3D-glut/cmake_install.cmake")
  include("C:/Users/Donato/Documents/git/opencal/build/OpenCAL/examples/mod2CA3D-glut/cmake_install.cmake")
  include("C:/Users/Donato/Documents/git/opencal/build/OpenCAL/examples/sciara-fv2/cmake_install.cmake")
  include("C:/Users/Donato/Documents/git/opencal/build/OpenCAL/examples/sciddicaS3hex-unsafe-explicit-glut/cmake_install.cmake")
  include("C:/Users/Donato/Documents/git/opencal/build/OpenCAL/examples/sciddicaS3hex-unsafe-glut/cmake_install.cmake")
  include("C:/Users/Donato/Documents/git/opencal/build/OpenCAL/examples/sciddicaT/cmake_install.cmake")
  include("C:/Users/Donato/Documents/git/opencal/build/OpenCAL/examples/sciddicaT-activecells-glut/cmake_install.cmake")
  include("C:/Users/Donato/Documents/git/opencal/build/OpenCAL/examples/sciddicaT-glut/cmake_install.cmake")
  include("C:/Users/Donato/Documents/git/opencal/build/OpenCAL/examples/sciddicaT-unsafe/cmake_install.cmake")
  include("C:/Users/Donato/Documents/git/opencal/build/OpenCAL/examples/sciddicaT-unsafe-glut/cmake_install.cmake")

endif()


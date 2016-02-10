#.rst:
# FindOpenCAL
# -----------
#
# This module looks for OpenCAL
#
# OpenCAL is a library for Cellular Automata .  Please see
# https://github.com/OpenCALTeam/opencal
#
# This module accepts the following optional variables:
#   OPENCAL_HOME - OpenCAL Source Root
#
# This module reports information about the OpenCAL installation in
# several variables.  General variables::
#
#   OPENCAL_VERSION - OPENCAL release version
#   OPENCAL_FOUND - true if the includes and libraries were found
#   OPENCAL_LIBRARIES - found libraries
#   OPENCAL_INCLUDE_DIRS - the directories containing the OPENCAL headers
#
# -----------LICENSE---------------
#
# (C) Copyright University of Calabria and others.
# All rights reserved. This program and the accompanying materials
# are made available under the terms of the GNU Lesser General Public License
# (LGPL) version 2.1 which accompanies this distribution, and is available at
# http://www.gnu.org/licenses/lgpl-2.1.html
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.


function(_OPENCAL_FIND)
  # Released versions of OPENCAL, including generic short forms
  set(opencal_versions
      1.0.0
      1.0
      1.1
      0.1)
  list(APPEND opencal_versions ${})
# Set up search paths, taking compiler into account.  Search Ice_HOME,
# with ICE_HOME in the environment as a fallback if unset.
if(OPENCAL_HOME)
  list(APPEND OPENCAL_ROOTS "${OPENCAL_HOME}")
else()
  if(NOT "$ENV{OPENCAL_HOME}" STREQUAL "")
    file(TO_CMAKE_PATH "$ENV{OPENCAL_HOME}" NATIVE_PATH)
    list(APPEND OPENCAL_ROOTS "${}")
    set(OPENCAL_HOME "${NATIVE_PATH}"
    CACHE PATH "Location of the OpenCAL installation" FORCE)
  endif()
endif()



  list(APPEND OPENCAL_ROOTS "/usr/local/")



if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  # 64-bit path suffix
  set(_x64 "/x64")
  # 64-bit library directory
  set(_lib64 "lib64")
endif()
#library suffixes
list(APPEND opencal_library_suffixes "${_lib64}" "lib${_x64}" "lib")
#include suffixes
list(APPEND opencal_include_suffixes "include/")
#add search path for each version released
foreach(opencal_version ${opencal_versions})
  list(APPEND opencal_include_suffixes "include/OpenCAL-${opencal_version}/")
  list(APPEND opencal_include_suffixes "include/OpenCAL-${opencal_version}/")
endforeach()

if(OPENCAL_DEBUG)
  message(STATUS "--------FindOpenCAL.cmake search debug--------")
  message(STATUS "OPENCAL_DEBUG binary path search order: ${OPENCAL_ROOTS}")
  message(STATUS "OPENCAL_DEBUG include  suffixes order: ${opencal_include_suffixes}")
  message(STATUS "OPENCAL_DEBUG library suffixes order: ${opencal_library_suffixes}")
  message(STATUS "----------------")
endif()

find_path(OPENCAL_INCLUDE_DIR
          NAMES
              "cal2D.h"
          HINTS
              ${OPENCAL_ROOTS}
          PATH_SUFFIXES
              ${opencal_include_suffixes}
          DOC
              "OpenCAL include directory")

set(OPENCAL_INCLUDE_DIRS ${OPENCAL_INCLUDE_DIR} PARENT_SCOPE)

find_library(OPENCAL_LIBRARIES
    NAMES
        opencal libopencal
    HINTS
        ${OPENCAL_ROOTS}
    PATH_SUFFIXES
        ${opencal_library_suffixes}
        )

  set(OPENCAL_LIBRARIES ${OPENCAL_LIBRARIES} PARENT_SCOPE)



if(OPENCAL_DEBUG)
  message(STATUS "--------FindOpenCAL.cmake result debug--------")
    message(STATUS "OpenCAL_VERSION number: ${OPENCAL_VERSION}")
    message(STATUS "OpenCAL_HOME directory: ${OPENCAL_HOME}")
    message(STATUS "OpenCAL_INCLUDE_DIRS directory: ${OPENCAL_INCLUDE_DIR}")
    message(STATUS "OpenCAL_LIBRARIES: ${OPENCAL_LIBRARIES}")
  message(STATUS "----------------")
endif()

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(OPENCAL
                                  FOUND_VAR OPENCAL_FOUND
                                  REQUIRED_VARS OPENCAL_INCLUDE_DIR
                                                OPENCAL_LIBRARIES
                                  VERSION_VAR OPENCAL_VERSION
                                FAIL_MESSAGE "Failed to find all OPENCAL components")


endfunction()

_OPENCAL_FIND()

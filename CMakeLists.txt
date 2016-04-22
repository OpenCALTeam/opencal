# opencal/CMakeLists.txt
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
cmake_policy(SET CMP0048 NEW)

include(configure.cmake)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
project(OpenCAL-${OPENCAL_VERSION}
		VERSION ${OPENCAL_VERSION}
		LANGUAGES C CXX)


set(INSTALL_ROOT_DIR opencal-${OPENCAL_VERSION})
set(INSTALL_INCLUDE_DIR  "${INSTALL_ROOT_DIR}/include")
set(INSTALL_LIB_DIR "${INSTALL_ROOT_DIR}/lib" )
set(INSTALL_BIN_DIR "${INSTALL_ROOT_DIR}/bin")
set(INSTALL_DOCS_DIR "${INSTALL_ROOT_DIR}/doc")


add_definitions( -DPACKAGE="OPENCAL" -DVERSION="${OPENCAL_VERSION}" )

if(BUILD_DOCUMENTATION)
	find_package(Doxygen)
endif(BUILD_DOCUMENTATION)

if (BUILD_OPENCAL_SERIAL)
	add_subdirectory(OpenCAL)
endif(BUILD_OPENCAL_SERIAL)

if(BUILD_OPENCAL_OMP)
	add_subdirectory(OpenCAL-OMP)
endif(BUILD_OPENCAL_OMP)

if(BUILD_OPENCAL_CL)
	add_subdirectory(OpenCAL-CL)
endif(BUILD_OPENCAL_CL)

if(BUILD_OPENCAL_GL)
	add_subdirectory(OpenCAL-GL)
endif(BUILD_OPENCAL_GL)

if(BUILD_OPENCAL_PP)
	add_subdirectory(OpenCAL++)
endif(BUILD_OPENCAL_PP)

if(BUILD_OPENCAL_TESTS)
	add_subdirectory(tests)
endif(BUILD_OPENCAL_TESTS)

if(BUILD_OPENCAL_CL)
	string(ASCII 27 Esc)
	set(Red         "${Esc}[31m")
	set(ColourReset "${Esc}[m")
	set(ColourBold  "${Esc}[1m")
	get_filename_component(KERNELPATH "${CMAKE_INSTALL_PREFIX}/${INSTALL_ROOT_DIR}/include/OpenCAL-CL/kernel/" ABSOLUTE)
	MESSAGE(STATUS "${Red}AFTER INSTALLATION, PLEASE EXPORT THE FOLLOWING ENVIRONMENT VARIABLE...${ColourReset}")
	MESSAGE(STATUS "${ColourBold}${Red}export OPENCALCL_PATH=${KERNELPATH}${ColourReset}")
endif(BUILD_OPENCAL_CL)


# uninstall target
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
    IMMEDIATE @ONLY)

add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)

# opencal/CMakeLists.txt
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
cmake_policy(SET CMP0048 NEW)

include(configure.cmake)

project(OpenCAL-ALL 
		VERSION ${OPENCAL_VERSION}
		LANGUAGES C CXX)


if ( ENABLE_SHARED )
	set( SHAREDORSTATIC SHARED )
else()
	set( SHAREDORSTATIC STATIC )
endif()

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
	get_filename_component(KERNELPATH "${CMAKE_INSTALL_PREFIX}/${INSTALL_SHARE_DIR}/OpenCAL-CL/" ABSOLUTE)
	MESSAGE(STATUS "${Red}(*AFTER MAKE INSTALL*) PLEASE EXPORT THE FOLLOWING ENVIRONMENT VARIABLE...${ColourReset}")
	MESSAGE(STATUS "${ColourBold}${Red}export OPENCALCL_PATH=${KERNELPATH}${ColourReset}")
endif(BUILD_OPENCAL_CL)


# uninstall target
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
    IMMEDIATE @ONLY)

add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)

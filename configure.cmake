
set(INSTALL_BIN_DIR bin CACHE PATH "Installation directory for executables")
set(INSTALL_LIB_DIR lib CACHE PATH "Installation directory for libraries")
set(INSTALL_INCLUDE_DIR include CACHE PATH "Installation directory for headers")
set(INSTALL_SHARE_DIR share CACHE PATH "Installation directory for data files")
set(INSTALL_EXAMPLE_DIR share/OpenCAL-Examples CACHE PATH "Installation directory for data files")


set(OPENCAL_MAJOR_VERSION 1)
set(OPENCAL_MINOR_VERSION 0)
set(OPENCAL_PATCH_VERSION 0)
set(OPENCAL_VERSION
  ${OPENCAL_MAJOR_VERSION}.${OPENCAL_MINOR_VERSION}.${OPENCAL_PATCH_VERSION})


option(BUILD_DOCUMENTATION  "Build the HTML based API documentation (Doxygen required)"             OFF)
option(BUILD_OPENCAL_SERIAL "Build the OpenCAL serial version"                                      ON )
option(BUILD_OPENCAL_OMP    "Build the OpenCAL-OMP OpenMP parallel version (OpenMP required)"       OFF)
option(BUILD_OPENCAL_OMP_PARALLEL  "Controls if OpenCAL-OMP compiled agaist libomp. If off just one processor is used."		 				ON)
option(BUILD_OPENCAL_CL     "Build the OpenCAL-CL OpenCL parallel version (OpenCL required)"        OFF)
option(BUILD_OPENCAL_GL     "Build the OpenCAL-GL visualization library (OpenGL and GLUT required)" OFF)
option(BUILD_EXAMPLES       "Build the examples for each OpenCAL version"                           OFF)
option(BUILD_OPENCAL_PP     "Build the OpenCAL-C++ version (C++11 Required)"		 				OFF)
option(BUILD_OPENCAL_TESTS  "Build the test suite of application for OPENCAL"		 				OFF)


option(ENABLE_SHARED "Enable Shared Libraries" OFF)

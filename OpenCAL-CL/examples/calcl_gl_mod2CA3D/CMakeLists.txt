# opencal/OpenCAL-CL/examples/calcl_gl_mod2CA3D/CMakeLists.txt
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

find_package(OpenCL REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(Threads REQUIRED)
find_library(M_LIB m)

include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}/include
	${PROJECT_SOURCE_DIR}/OpenCAL/include
	${PROJECT_SOURCE_DIR}/OpenCAL-GL/include
	${PROJECT_SOURCE_DIR}/OpenCAL-CL/include
	${PROJECT_SOURCE_DIR}/OpenCAL-CL/OpenCL-Utils/include
	${OPENGL_INCLUDE_DIRS}
	${GLUT_INCLUDE_DIR}
)
include_directories( ${OpenCL_INCLUDE_DIRS} )

link_directories(${OpenCL_LIBRARIES})

link_directories(
	${CMAKE_CURRENT_SOURCE_DIR}/source
	${PROJECT_SOURCE_DIR}/OpenCAL/lib
	${PROJECT_SOURCE_DIR}/OpenCAL-GL/lib
)
LINK_DIRECTORIES(${PROJECT_SOURCE_DIR}/OpenCAL-CL/lib)
LINK_DIRECTORIES(${PROJECT_SOURCE_DIR}/OpenCAL-CL/OpenCL-Utils/lib)

file(GLOB SRC
    "include/*.h"
    "source/*.cpp"
    "source/*.c"
)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

add_executable(calcl_gl_mod2CA3D ${SRC})
target_link_libraries(calcl_gl_mod2CA3D opencal-gl opencal-cl opencl_utils opencal ${M_LIB} ${OPENGL_LIBRARIES} ${GLUT_glut_LIBRARY} -lpthread ${OpenCL_LIBRARY})

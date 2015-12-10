# opencal/OpenCAL-CL/examples/calcl_mod2CA3D/CMakeLists.txt
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

find_package( OpenCL REQUIRED )
include_directories( ${OpenCL_INCLUDE_DIRS} )
link_directories(${OpenCL_LIBRARIES})

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/source)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

LINK_DIRECTORIES(${PROJECT_SOURCE_DIR}/OpenCAL/lib)
LINK_DIRECTORIES(${PROJECT_SOURCE_DIR}/OpenCAL-CL/lib)
LINK_DIRECTORIES(${PROJECT_SOURCE_DIR}/OpenCAL-CL/OpenCL-Utils/lib)

file(GLOB SRC
	"include/*.h"
    "source/*.c"
)

add_executable(calcl_mod2CA3D ${SRC})
target_link_libraries(calcl_mod2CA3D opencal-cl opencal opencl_utils m ${OpenCL_LIBRARY})

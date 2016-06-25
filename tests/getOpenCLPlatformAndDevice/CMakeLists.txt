# opencal/OpenCAL-CL/examples/sciddicaT-calcl/CMakeLists.txt
cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
find_package(OpenCAL 1.0 COMPONENTS  OpenCAL  OpenCAL-CL REQUIRED)

find_package( OpenCL REQUIRED )
find_library(M_LIB m)
include_directories( ${OpenCL_INCLUDE_DIRS}
${OPENCAL_INCLUDE_DIR})
link_directories(${OpenCL_LIBRARIES})

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/source)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include
                    ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
include_directories(${PROJECT_SOURCE_DIR}/OpenCAL/include)
include_directories(${PROJECT_SOURCE_DIR}/OpenCAL-CL/include)
include_directories(${PROJECT_SOURCE_DIR}/OpenCAL-CL/OpenCL-Utils/include)


LINK_DIRECTORIES(${PROJECT_SOURCE_DIR}/OpenCAL/lib)
LINK_DIRECTORIES(${PROJECT_SOURCE_DIR}/OpenCAL-CL/lib)
LINK_DIRECTORIES(${PROJECT_SOURCE_DIR}/OpenCAL-CL/OpenCL-Utils/lib)


file(GLOB SRC
    "include/*.h"
    "source/*.c"
)

add_executable(calcl-getPlatformAndDevice-test ${SRC})
target_link_libraries(calcl-getPlatformAndDevice-test  opencal-cl ${M_LIB} opencal ${OpenCL_LIBRARY} ${OPENCAL_LIBRARIES} )

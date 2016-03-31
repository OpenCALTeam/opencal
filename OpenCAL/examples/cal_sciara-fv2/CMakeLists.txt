# opencal/OpenCAL/examples/cal_sciara-fv2/CMakeLists.txt
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

#resolve the required dependencies
find_package(OpenCAL 1.0 COMPONENTS  OpenCAL  REQUIRED)


include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include
                    ${OPENCAL_INCLUDE_DIR}
                    )

file(GLOB SRC
    "include/*.h"
    "source/*.c"
	  "source/*.cpp"
)

add_executable(cal_sciara_fv2 ${SRC})
target_link_libraries(cal_sciara_fv2 ${OPENCAL_LIBRARIES})

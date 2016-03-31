# opencal/OpenCAL/examples/cal_mod2CA3D/CMakeLists.txt
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

find_package(OpenCAL 1.0 COMPONENTS  OpenCAL  REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include
                    ${OPENCAL_INCLUDE_DIR}
                    )


file(GLOB SRC
    "source/*.c"
)

add_executable(cal_mod2CA3D ${SRC})
target_link_libraries(cal_mod2CA3D ${OPENCAL_LIBRARIES})

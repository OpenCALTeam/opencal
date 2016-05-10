find_package(GTest REQUIRED)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

include_directories(${PROJECT_SOURCE_DIR}/OpenCAL++/include/)
include_directories(${GTEST_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/)


file(GLOB SRC
        "*.h"
        "*.cpp"
        )

message(STATUS  ${PROJECT_SOURCE_DIR}/OpenCAL++/include/ )
add_executable(opencal++_tests ${SRC} )
set_property(TARGET opencal++_tests PROPERTY CXX_STANDARD 14)
target_link_libraries(opencal++_tests ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES} pthread)
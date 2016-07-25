cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)


include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OPENGL_INCLUDE_DIRS}
        ${GLUT_INCLUDE_DIR}
        ${OPENCAL_INCLUDE_DIR}
)

file(GLOB SRC
    "include/*.h"
    "source/*.c"
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/source)

include_directories(${PROJECT_SOURCE_DIR}/OpenCAL-CPU/include)

add_executable(cal-cpu_sciddica-activecells-glut ${SRC})
target_link_libraries(cal-cpu_sciddica-activecells-glut opencal-cpu ${OPENGL_LIBRARIES} ${GLUT_glut_LIBRARY} )

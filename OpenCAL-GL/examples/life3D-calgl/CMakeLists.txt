# opencal/OpenCAL-GL/examples/life3D-calgl/CMakeLists.txt

cmake_minimum_required(VERSION 2.8 FATAL_ERROR)


find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(Threads REQUIRED)
find_library(M_LIB m)

if (${PTW32_STATIC} STREQUAL "ON")
	add_definitions(-DPTW32_STATIC_LIB)
	set(PTW32_STATIC_LIB ${PROJECT_SOURCE_DIR}/extern/pthread-win32/lib/x86/pthread_lib.lib)
	include_directories( ${PROJECT_SOURCE_DIR}/extern/pthread-win32/include )
endif()

if (${FREEGLUT_STATIC} STREQUAL "ON")
	add_definitions(-DFREEGLUT_STATIC)
endif()


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/source)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

LINK_DIRECTORIES(${PROJECT_SOURCE_DIR}/OpenCAL/lib)
LINK_DIRECTORIES(${PROJECT_SOURCE_DIR}/OpenCAL-GL/lib)

include_directories(${PROJECT_SOURCE_DIR}/OpenCAL/include)
include_directories(${PROJECT_SOURCE_DIR}/OpenCAL-GL/include)

include_directories( ${OPENGL_INCLUDE_DIRS}  ${GLUT_INCLUDE_DIR} )

file(GLOB SRC
    "include/*.h"
    "source/*.cpp"
    "source/*.c"
)

add_executable(life3D-calgl ${SRC})

if(WIN32)
  target_link_libraries(life3D-calgl opencal-gl opencal ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} ${CMAKE_THREAD_LIBS_INIT} ${PTW32_STATIC_LIB})
else(WIN32)
  target_link_libraries(life3D-calgl opencal-gl opencal ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} ${CMAKE_THREAD_LIBS_INIT} -lm)
endif(WIN32)

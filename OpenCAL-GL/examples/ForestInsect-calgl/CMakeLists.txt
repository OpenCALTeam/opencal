# opencal/OpenCAL-GL/examples/ForestInsect-calgl/CMakeLists.txt
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)


find_package(OpenGL  REQUIRED)
if (${USE_EXTERN_FREEGLUT_STATIC} STREQUAL "ON")
	set (GLUT_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/../extern/freeglut-2.8.1/include")
	set (GLUT_glut_LIBRARY "${PROJECT_SOURCE_DIR}/../extern/freeglut-2.8.1/lib/x86/freeglut_static.lib")
	add_definitions(-DFREEGLUT_STATIC)
else()
	find_package(GLUT REQUIRED)
endif()

if (${USE_EXTERN_PTW32_STATIC} STREQUAL "ON")
	set (PTW32_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/../extern/pthread-win32/include")
	set (PTW32_ptw32_LIBRARY "${PROJECT_SOURCE_DIR}/../extern/pthread-win32/lib/x86/pthread_lib.lib")
	add_definitions(-DPTW32_STATIC_LIB)
else()
	find_package(Threads REQUIRED)
endif()


FIND_PACKAGE(OpenMP)
IF(OPENMP_FOUND)
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
ENDIF()


include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}/include
	${PROJECT_SOURCE_DIR}/OpenCAL/include
	${PROJECT_SOURCE_DIR}/OpenCAL-GL/include
	${OPENGL_INCLUDE_DIRS}
	${GLUT_INCLUDE_DIR}
)


link_directories(
	${CMAKE_CURRENT_SOURCE_DIR}/source
	${PROJECT_SOURCE_DIR}/OpenCAL/lib
	${PROJECT_SOURCE_DIR}/OpenCAL-GL/lib
)


file(GLOB SRC
    "include/*.h"
    "source/*.cpp"
    "source/*.c"
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

add_executable(ForestInsect-calgl ${SRC})
if(WIN32)
	target_link_libraries(ForestInsect-calgl opencal-gl opencal ${OPENGL_LIBRARIES} ${GLUT_glut_LIBRARY}  ${PTW32_ptw32_LIBRARY})
else()
	target_link_libraries(ForestInsect-calgl -lm opencal-gl opencal ${OPENGL_LIBRARIES} ${GLUT_glut_LIBRARY} ${CMAKE_THREAD_LIBS_INIT})
endif()

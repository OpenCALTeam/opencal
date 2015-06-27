cmake_minimum_required(VERSION 2.8 FATAL_ERROR)


MACRO(SUBDIRLIST result curdir)
  FILE(GLOB children RELATIVE ${curdir} ${curdir}/*)
  SET(dirlist "")
  FOREACH(child ${children})
    IF(IS_DIRECTORY ${curdir}/${child})
        LIST(APPEND dirlist ${child})
    ENDIF()
  ENDFOREACH()
  SET(${result} ${dirlist})
ENDMACRO()

SUBDIRLIST(SUBDIRS ${CMAKE_CURRENT_SOURCE_DIR})

#add all subdirectory 
#FOREACH(subdir ${SUBDIRS})
#    ADD_SUBDIRECTORY(${subdir})
#ENDFOREACH()

ADD_SUBDIRECTORY(iso-3src-glut)
ADD_SUBDIRECTORY(life)
ADD_SUBDIRECTORY(life3D-glut)
ADD_SUBDIRECTORY(mod2CA3D-glut)
ADD_SUBDIRECTORY(sciara-fv2)
ADD_SUBDIRECTORY(sciddicaS3hex-unsafe-explicit-glut)
ADD_SUBDIRECTORY(sciddicaS3hex-unsafe-glut)
ADD_SUBDIRECTORY(sciddicaT)
ADD_SUBDIRECTORY(sciddicaT-activecells-glut)
ADD_SUBDIRECTORY(sciddicaT-glut)
ADD_SUBDIRECTORY(sciddicaT-unsafe)
ADD_SUBDIRECTORY(sciddicaT-unsafe-glut)

include ../Makefile.inc


#INC_PATH 	= ./include
#SRC_PATH	= ./source
#BIN_PATH	= ./bin
#OPENCAL_LIB_PATH	= ../OpenCAL/lib
#OPENCAL_INC_PATH	= ../OpenCAL/include
#OPTFLAG		= -g

all: 
	LD_RUN_PATH=../OpenCAL-OMP/build/lib gcc -o sciddicaT source/sciddicaT.c  ../OpenCAL-GL/build/lib/libopencalgl.a -I../OpenCAL-OMP/build/include/ -I../OpenCAL-GL/build/include/ -L../OpenCAL-OMP/build/lib/ -lGL -lGLU -lglut -lm -lpthread -fopenmp -lopencal


clean:
	@$(RM) sciddicaT *.o

include ../Makefile.inc


#INC_PATH 	= ./include
CCP = g++
SRC_PATH				= ./source
BIN_PATH				= ./bin
OPENCAL_LIB_PATH		= ../OpenCAL/lib
OPENCAL_INC_PATH		= ../OpenCAL/include
OPENCAL-CL_LIB_PATH		= ../OpenCAL-CL/lib
OPENCAL-CL_INC_PATH		= ../OpenCAL-CL/include
OPENCL_UTILS_PATH 		= ../OpenCAL-CL/OpenCL-Utils
OPENCL_UTILS_LIB_PATH	= $(OPENCL_UTILS_PATH)/lib
OPENCL_UTILS_INC_PATH	= $(OPENCL_UTILS_PATH)/include
OPENCL_INC_PATH			= /usr/local/cuda-5.5/include


all: dir $(BIN_PATH)/calclsciddicaT

dir:
	mkdir -p ./bin 

$(BIN_PATH)/calclsciddicaT: $(BIN_PATH)/sciddicaT.o $(BIN_PATH)/gui.o
	$(CC) -o $(BIN_PATH)/calclsciddicaT $(BIN_PATH)/sciddicaT.o $(BIN_PATH)/gui.o $(OPENCL_UTILS_LIB_PATH)/libopencl_utils.a $(OPENCAL_LIB_PATH)/opencal.a $(OPENCAL-CL_LIB_PATH)/libopencal-cl.a -lm -lOpenCL 
	ln -fs $(BIN_PATH)/calclsciddicaT .

$(BIN_PATH)/gui.o: $(SRC_PATH)/gui.c
	$(CC) -c $(SRC_PATH)/gui.c $(CFLAGS)  -I$(OPENCL_UTILS_INC_PATH) -I$(OPENCAL_INC_PATH) -I$(OPENCAL-CL_INC_PATH) -I$(OPENCL_INC_PATH) -o $(BIN_PATH)/gui.o

$(BIN_PATH)/sciddicaT.o: $(SRC_PATH)/sciddicaT.c
	$(CC) -c $(SRC_PATH)/sciddicaT.c $(CFLAGS) -I$(OPENCAL_INC_PATH) -I$(OPENCAL-CL_INC_PATH) -I$(OPENCL_UTILS_INC_PATH) -I$(OPENCL_INC_PATH) -o $(BIN_PATH)/sciddicaT.o

clean:
	@$(RM) calclsciddicaT $(BIN_PATH)/*

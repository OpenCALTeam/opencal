include ../../../Makefile.inc

CCP =g++
INC_PATH 				= ./include
SRC_PATH				= ./source
BIN_PATH				= ./bin
OPENCAL_LIB_PATH		= ../../../OpenCAL/lib
OPENCAL_INC_PATH		= ../../../OpenCAL/include
OPENCAL-CL_LIB_PATH		= ../../../OpenCAL-CL/lib
OPENCAL-CL_INC_PATH		= ../../../OpenCAL-CL/include
OPENCL_UTILS_PATH 		= ../../../OpenCAL-CL/OpenCL-Utils
OPENCL_UTILS_LIB_PATH	= $(OPENCL_UTILS_PATH)/lib
OPENCL_UTILS_INC_PATH	= $(OPENCL_UTILS_PATH)/include
OPENCL_INC_PATH			= $(value OPENCL_INCLUDE_PATH)

all: dir $(BIN_PATH)/calclmbusu

dir:
	mkdir -p ./bin 

$(BIN_PATH)/calclmbusu: $(BIN_PATH)/main.o $(BIN_PATH)/MbusuCL.o
	$(CCP) -o $(BIN_PATH)/calclmbusu $(BIN_PATH)/MbusuCL.o   $(BIN_PATH)/main.o $(OPENCL_UTILS_LIB_PATH)/libopencl_utils.a $(OPENCAL_LIB_PATH)/opencal.a $(OPENCAL-CL_LIB_PATH)/libopencal-cl.a -lm -lOpenCL  
	ln -fs $(BIN_PATH)/calclmbusu .

$(BIN_PATH)/main.o: $(SRC_PATH)/main.cpp
	$(CCP) -c $(SRC_PATH)/main.cpp $(CFLAGS)  -I$(OPENCL_UTILS_INC_PATH) -I$(OPENCAL_INC_PATH) -I$(INC_PATH) -I$(OPENCAL-CL_INC_PATH) -I$(OPENCL_INC_PATH) -o $(BIN_PATH)/main.o 

$(BIN_PATH)/MbusuCL.o: $(SRC_PATH)/MbusuCL.cpp
	$(CCP) -c $(SRC_PATH)/MbusuCL.cpp $(CFLAGS)  -I$(OPENCL_UTILS_INC_PATH) -I$(OPENCAL_INC_PATH) -I$(INC_PATH) -I$(OPENCAL-CL_INC_PATH) -I$(OPENCL_INC_PATH) -o  $(BIN_PATH)/MbusuCL.o 

clean:
	@$(RM) calclmbusu $(BIN_PATH)/* 
	rm -fr ./bin
	
	
	
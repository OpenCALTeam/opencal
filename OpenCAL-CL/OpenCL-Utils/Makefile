include ../Makefile.inc

INC_PATH 	= ./include
SRC_PATH	= ./source
BIN_PATH	= ./bin
LIB_PATH	= ./lib

all: dir $(LIB_PATH)/libopencl_utils.a

dir:
	mkdir -p ./bin 
	mkdir -p ./lib 
	
$(LIB_PATH)/libopencl_utils.a: $(BIN_PATH)/OpenCL_Utility.o
	ar rcs  $(LIB_PATH)/libopencl_utils.a $(BIN_PATH)/OpenCL_Utility.o 

$(BIN_PATH)/OpenCL_Utility.o: $(SRC_PATH)/OpenCL_Utility.c
	$(CC) -c $(SRC_PATH)/OpenCL_Utility.c $(CFLAGS) -I$(OPENCL_INC_PATH) -I$(INC_PATH) -o $(BIN_PATH)/OpenCL_Utility.o
	
clean:
	rm -fr bin
	rm -fr lib
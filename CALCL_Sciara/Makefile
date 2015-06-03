include ../Makefile.inc

CCP = g++
INC_PATH 				= ./include
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
all: dir $(BIN_PATH)/sciaraCL

dir:
	mkdir -p ./bin 

$(BIN_PATH)/sciaraCL: $(BIN_PATH)/main.o $(BIN_PATH)/Sciara.o $(BIN_PATH)/GISInfo.o $(BIN_PATH)/configurationPathLib.o $(BIN_PATH)/vent.o $(BIN_PATH)/io.o
	$(CCP) -o $(BIN_PATH)/sciaraCL $(BIN_PATH)/main.o $(BIN_PATH)/Sciara.o $(BIN_PATH)/GISInfo.o  $(BIN_PATH)/configurationPathLib.o $(BIN_PATH)/vent.o $(BIN_PATH)/io.o -L$(OPENCL_UTILS_LIB_PATH) -lopencl_utils $(OPENCAL_LIB_PATH)/opencal.a -L$(OPENCAL-CL_LIB_PATH) -lopencal-cl -lOpenCL
	ln -fs $(BIN_PATH)/sciaraCL .

$(BIN_PATH)/main.o: $(SRC_PATH)/main.cpp
	$(CCP) -c $(SRC_PATH)/main.cpp $(CFLAGS) -I$(OPENCAL_INC_PATH) -I$(OPENCAL-CL_INC_PATH) -I$(INC_PATH) -I$(OPENCL_UTILS_INC_PATH) -I$(OPENCL_INC_PATH) -o $(BIN_PATH)/main.o

$(BIN_PATH)/Sciara.o: $(SRC_PATH)/Sciara.cpp
	$(CCP) -c $(SRC_PATH)/Sciara.cpp $(CFLAGS) -I$(OPENCAL_INC_PATH) -I$(OPENCAL-CL_INC_PATH) -I$(INC_PATH) -I$(OPENCL_UTILS_INC_PATH) -I$(OPENCL_INC_PATH) -o $(BIN_PATH)/Sciara.o

$(BIN_PATH)/GISInfo.o: $(SRC_PATH)/GISInfo.cpp
	$(CCP) -c $(SRC_PATH)/GISInfo.cpp $(CFLAGS) -I$(OPENCAL_INC_PATH) -I$(OPENCAL-CL_INC_PATH) -I$(INC_PATH) -I$(OPENCL_UTILS_INC_PATH) -I$(OPENCL_INC_PATH) -o $(BIN_PATH)/GISInfo.o

$(BIN_PATH)/configurationPathLib.o: $(SRC_PATH)/configurationPathLib.cpp
	$(CCP) -c $(SRC_PATH)/configurationPathLib.cpp $(CFLAGS) -I$(OPENCAL_INC_PATH) -I$(OPENCAL-CL_INC_PATH) -I$(INC_PATH) -I$(OPENCL_UTILS_INC_PATH) -I$(OPENCL_INC_PATH) -o $(BIN_PATH)/configurationPathLib.o

$(BIN_PATH)/vent.o: $(SRC_PATH)/vent.cpp
	$(CCP) -c $(SRC_PATH)/vent.cpp $(CFLAGS) -I$(OPENCAL_INC_PATH) -I$(OPENCAL-CL_INC_PATH) -I$(INC_PATH) -I$(OPENCL_UTILS_INC_PATH) -I$(OPENCL_INC_PATH) -o $(BIN_PATH)/vent.o

$(BIN_PATH)/io.o: $(SRC_PATH)/io.cpp
	$(CCP) -c $(SRC_PATH)/io.cpp $(CFLAGS) -I$(OPENCAL_INC_PATH) -I$(OPENCAL-CL_INC_PATH) -I$(INC_PATH) -I$(OPENCL_UTILS_INC_PATH) -I$(OPENCL_INC_PATH) -o $(BIN_PATH)/io.o

clean:
	@$(RM) sciara $(BIN_PATH)/*
	rm -fr ./bin
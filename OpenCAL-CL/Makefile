include ./Makefile.inc


INC_PATH 	= ./include
SRC_PATH	= ./source
BIN_PATH	= ./bin
LIB_PATH	= ./lib


all: dir utils $(LIB_PATH)/libopencal-cl.a

utils:
	cd $(OPENCL_UTILS_PATH) && $(MAKE) clean && $(MAKE)

dir:
	mkdir -p $(BIN_PATH)
	mkdir -p $(LIB_PATH)

$(LIB_PATH)/libopencal-cl.a: $(BIN_PATH)/calcl2D.o $(BIN_PATH)/calcl3D.o $(OPENCAL_LIB_PATH)/opencal.a
	ar rcs $(LIB_PATH)/libopencal-cl.a $(BIN_PATH)/calcl2D.o $(BIN_PATH)/calcl3D.o

$(BIN_PATH)/calcl2D.o: $(SRC_PATH)/calcl2D.c
	$(CC) -c $(SRC_PATH)/calcl2D.c $(CFLAGS) -I$(INC_PATH) -I$(OPENCL_UTILS_INC_PATH) -I$(OPENCAL_INC_PATH) -I$(OPENCL_INC_PATH) -o $(BIN_PATH)/calcl2D.o

$(BIN_PATH)/calcl3D.o: $(SRC_PATH)/calcl3D.c
	$(CC) -c $(SRC_PATH)/calcl3D.c $(CFLAGS) -I$(INC_PATH) -I$(OPENCL_UTILS_INC_PATH) -I$(OPENCAL_INC_PATH) -I$(OPENCL_INC_PATH) -o $(BIN_PATH)/calcl3D.o

	
clean:
	rm -fr $(BIN_PATH)
	rm -fr $(LIB_PATH)

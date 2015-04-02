include ../Makefile.inc


INC_PATH 			= ./include
SRC_PATH			= ./source
BIN_PATH			= ./bin
LIB_PATH			= ./lib
OPENCAL_INC_PATH 	= ../OpenCAL/include
#OPTFLAG			= -g

all: $(LIB_PATH)/opencal-gl.a

$(LIB_PATH)/opencal-gl.a: $(BIN_PATH)/calgl2D.o 
	ar rcs $(LIB_PATH)/opencal-gl.a $(BIN_PATH)/calgl2D.o

$(BIN_PATH)/calglModelViewParameter.o: $(SRC_PATH)/calglModelViewParameter.c
	$(CC) -c $(SRC_PATH)/calglModelViewParameter.c $(CFLAGS) -I$(INC_PATH) -I$(OPENCAL_INC_PATH) -o $(BIN_PATH)/calglModelViewParameter.o

$(BIN_PATH)/calglUtils.o: $(SRC_PATH)/calglUtils.c
	$(CC) -c $(SRC_PATH)/calglUtils.c $(CFLAGS) -I$(INC_PATH) -I$(OPENCAL_INC_PATH) -o $(BIN_PATH)/calglUtils.o

clean:
	@$(RM) $(LIB_PATH)/opencal-gl.a $(BIN_PATH)/*

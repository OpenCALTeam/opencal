include ../Makefile.inc


#INC_PATH 	= ./include
SRC_PATH	= ./source
BIN_PATH	= ./bin
OPENCAL_LIB_PATH	= ../OpenCAL/lib
OPENCAL_INC_PATH	= ../OpenCAL/include/
#OPTFLAG		= -g

all: $(BIN_PATH)/sciddicaS3Hex

$(BIN_PATH)/sciddicaS3Hex: $(BIN_PATH)/sciddicaS3Hex.o $(BIN_PATH)/gui.o $(OPENCAL_LIB_PATH)/opencal.a
	$(CC) -o $(BIN_PATH)/sciddicaS3Hex $(BIN_PATH)/sciddicaS3Hex.o $(BIN_PATH)/gui.o $(OPENCAL_LIB_PATH)/opencal.a -lGL -lGLU -lglut
	ln -fs $(BIN_PATH)/sciddicaS3Hex .

$(BIN_PATH)/gui.o: $(SRC_PATH)/gui.c
	$(CC) -c $(SRC_PATH)/gui.c $(CFLAGS) -I$(OPENCAL_INC_PATH) -o $(BIN_PATH)/gui.o

$(BIN_PATH)/sciddicaS3Hex.o: $(SRC_PATH)/sciddicaS3Hex.c
	$(CC) -c $(SRC_PATH)/sciddicaS3Hex.c $(CFLAGS) -I$(OPENCAL_INC_PATH) -o $(BIN_PATH)/sciddicaS3Hex.o

clean:
	@$(RM) sciddicaS3Hex $(BIN_PATH)/*.o $(BIN_PATH)/sciddicaS3Hex

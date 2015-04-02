include ../Makefile.inc


#INC_PATH 	= ./include
SRC_PATH	= ./source
BIN_PATH	= ./bin
OPENCAL_LIB_PATH	= ../OpenCAL/lib
OPENCAL_INC_PATH	= ../OpenCAL/include
#OPTFLAG		= -g

all: $(BIN_PATH)/sciddicaT

$(BIN_PATH)/sciddicaT: $(BIN_PATH)/sciddicaT.o $(OPENCAL_LIB_PATH)/opencal.a
	$(CC) -o $(BIN_PATH)/sciddicaT $(BIN_PATH)/sciddicaT.o $(OPENCAL_LIB_PATH)/opencal.a
	ln -fs $(BIN_PATH)/sciddicaT .

$(BIN_PATH)/sciddicaT.o: $(SRC_PATH)/sciddicaT.c
	$(CC) -c $(SRC_PATH)/sciddicaT.c $(CFLAGS) -I$(OPENCAL_INC_PATH) -o $(BIN_PATH)/sciddicaT.o

clean:
	@$(RM) sciddicaT $(BIN_PATH)/*

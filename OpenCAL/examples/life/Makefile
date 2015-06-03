include ../Makefile.inc


#INC_PATH 	= ./include
SRC_PATH	= ./source
BIN_PATH	= ./bin
OPENCAL_LIB_PATH	= ../OpenCAL/lib
OPENCAL_INC_PATH	= ../OpenCAL/include
#OPTFLAG		= -g

all: dir $(BIN_PATH)/life

dir:
	mkdir -p ./bin

$(BIN_PATH)/life: $(BIN_PATH)/life.o $(OPENCAL_LIB_PATH)/opencal.a
	$(CC) -o $(BIN_PATH)/life $(BIN_PATH)/life.o $(OPENCAL_LIB_PATH)/opencal.a
	ln -fs $(BIN_PATH)/life .

$(BIN_PATH)/life.o: $(SRC_PATH)/life.c
	$(CC) -c $(SRC_PATH)/life.c $(CFLAGS) -I$(OPENCAL_INC_PATH) -o $(BIN_PATH)/life.o

clean:
	@$(RM) life $(BIN_PATH)/*
	rm -fr bin

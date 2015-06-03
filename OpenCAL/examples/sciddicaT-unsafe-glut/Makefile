include ../Makefile.inc


#INC_PATH 	= ./include
SRC_PATH	= ./source
BIN_PATH	= ./bin
OPENCAL_LIB_PATH	= ../OpenCAL/lib
OPENCAL_INC_PATH	= ../OpenCAL/include/
#OPTFLAG		= -g

all:dir $(BIN_PATH)/sciddicaT
dir:
	mkdir -p ./bin


$(BIN_PATH)/sciddicaT: $(BIN_PATH)/sciddicaTunsafe.o $(BIN_PATH)/gui.o $(OPENCAL_LIB_PATH)/opencal.a
	$(CC) -o $(BIN_PATH)/sciddicaT $(BIN_PATH)/sciddicaTunsafe.o $(BIN_PATH)/gui.o $(OPENCAL_LIB_PATH)/opencal.a -lGL -lGLU -lglut
	ln -fs $(BIN_PATH)/sciddicaT .

$(BIN_PATH)/gui.o: $(SRC_PATH)/gui.c
	$(CC) -c $(SRC_PATH)/gui.c $(CFLAGS) -I$(OPENCAL_INC_PATH) -o $(BIN_PATH)/gui.o

$(BIN_PATH)/sciddicaTunsafe.o: $(SRC_PATH)/sciddicaTunsafe.c
	$(CC) -c $(SRC_PATH)/sciddicaTunsafe.c $(CFLAGS) -I$(OPENCAL_INC_PATH) -o $(BIN_PATH)/sciddicaTunsafe.o

clean:
	@$(RM) sciddicaT $(BIN_PATH)/*.o $(BIN_PATH)/sciddicaT
	rm -fr bin

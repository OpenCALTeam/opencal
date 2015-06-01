include ../Makefile.inc


#INC_PATH 	= ./include
SRC_PATH	= ./source
BIN_PATH	= ./bin
OPENCAL_LIB_PATH	= ../OpenCAL/lib
OPENCAL_INC_PATH	= ../OpenCAL/include
#OPTFLAG		= -g

all:dir $(BIN_PATH)/life3D-glut

dir:
	mkdir -p ./bin

$(BIN_PATH)/life3D-glut: $(BIN_PATH)/life3D.o $(BIN_PATH)/gui.o $(OPENCAL_LIB_PATH)/opencal.a
	$(CC) -o $(BIN_PATH)/life3D-glut $(BIN_PATH)/life3D.o $(BIN_PATH)/gui.o $(OPENCAL_LIB_PATH)/opencal.a -lGL -lGLU -lglut
	ln -fs $(BIN_PATH)/life3D-glut .

$(BIN_PATH)/gui.o: $(SRC_PATH)/gui.c
	$(CC) -c $(SRC_PATH)/gui.c $(CFLAGS) -I$(OPENCAL_INC_PATH) -o $(BIN_PATH)/gui.o

$(BIN_PATH)/life3D.o: $(SRC_PATH)/life3D.c
	$(CC) -c $(SRC_PATH)/life3D.c $(CFLAGS) -I$(OPENCAL_INC_PATH) -o $(BIN_PATH)/life3D.o

clean:
	@$(RM) life3D-glut $(BIN_PATH)/*.o $(BIN_PATH)/life3D-glut
	rm -fr bin

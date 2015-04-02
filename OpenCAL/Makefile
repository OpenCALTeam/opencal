include ../Makefile.inc


INC_PATH 	= ./include
SRC_PATH	= ./source
BIN_PATH	= ./bin
LIB_PATH	= ./lib
#OPTFLAG		= -g

all: $(LIB_PATH)/opencal.a



$(LIB_PATH)/opencal.a:	$(BIN_PATH)/cal2D.o \
						$(BIN_PATH)/cal2DBuffer.o \
						$(BIN_PATH)/cal2DBufferIO.o \
						$(BIN_PATH)/cal2DIO.o \
						$(BIN_PATH)/cal2DRun.o \
						$(BIN_PATH)/cal2DUnsafe.o \
						$(BIN_PATH)/cal3D.o \
						$(BIN_PATH)/cal3DBuffer.o \
						$(BIN_PATH)/cal3DBufferIO.o \
						$(BIN_PATH)/cal3DIO.o \
						$(BIN_PATH)/cal3DRun.o \
						$(BIN_PATH)/cal3DUnsafe.o

	ar rcs $(LIB_PATH)/opencal.a \
	$(BIN_PATH)/cal2D.o \
	$(BIN_PATH)/cal2DBuffer.o \
	$(BIN_PATH)/cal2DBufferIO.o \
	$(BIN_PATH)/cal2DIO.o \
	$(BIN_PATH)/cal2DRun.o \
	$(BIN_PATH)/cal2DUnsafe.o \
	$(BIN_PATH)/cal3D.o \
	$(BIN_PATH)/cal3DBuffer.o \
	$(BIN_PATH)/cal3DBufferIO.o \
	$(BIN_PATH)/cal3DIO.o \
	$(BIN_PATH)/cal3DRun.o \
	$(BIN_PATH)/cal3DUnsafe.o



$(BIN_PATH)/cal2D.o: $(SRC_PATH)/cal2D.c
	$(CC) -c $(SRC_PATH)/cal2D.c $(CFLAGS) -I$(INC_PATH) -o $(BIN_PATH)/cal2D.o

$(BIN_PATH)/cal2DBuffer.o: $(SRC_PATH)/cal2DBuffer.c
	$(CC) -c $(SRC_PATH)/cal2DBuffer.c $(CFLAGS) -I$(INC_PATH) -o $(BIN_PATH)/cal2DBuffer.o

$(BIN_PATH)/cal2DBufferIO.o: $(SRC_PATH)/cal2DBufferIO.c
	$(CC) -c $(SRC_PATH)/cal2DBufferIO.c $(CFLAGS) -I$(INC_PATH) -o $(BIN_PATH)/cal2DBufferIO.o

$(BIN_PATH)/cal2DIO.o: $(SRC_PATH)/cal2DIO.c
	$(CC) -c $(SRC_PATH)/cal2DIO.c $(CFLAGS) -I$(INC_PATH) -o $(BIN_PATH)/cal2DIO.o

$(BIN_PATH)/cal2DRun.o: $(SRC_PATH)/cal2DRun.c
	$(CC) -c $(SRC_PATH)/cal2DRun.c $(CFLAGS) -I$(INC_PATH) -o $(BIN_PATH)/cal2DRun.o

$(BIN_PATH)/cal2DUnsafe.o: $(SRC_PATH)/cal2DUnsafe.c 
	$(CC) -c $(SRC_PATH)/cal2DUnsafe.c $(CFLAGS) -I$(INC_PATH) -o $(BIN_PATH)/cal2DUnsafe.o



$(BIN_PATH)/cal3D.o: $(SRC_PATH)/cal3D.c
	$(CC) -c $(SRC_PATH)/cal3D.c $(CFLAGS) -I$(INC_PATH) -o $(BIN_PATH)/cal3D.o

$(BIN_PATH)/cal3DBuffer.o: $(SRC_PATH)/cal3DBuffer.c
	$(CC) -c $(SRC_PATH)/cal3DBuffer.c $(CFLAGS) -I$(INC_PATH) -o $(BIN_PATH)/cal3DBuffer.o

$(BIN_PATH)/cal3DBufferIO.o: $(SRC_PATH)/cal3DBufferIO.c
	$(CC) -c $(SRC_PATH)/cal3DBufferIO.c $(CFLAGS) -I$(INC_PATH) -o $(BIN_PATH)/cal3DBufferIO.o

$(BIN_PATH)/cal3DIO.o: $(SRC_PATH)/cal3DIO.c
	$(CC) -c $(SRC_PATH)/cal3DIO.c $(CFLAGS) -I$(INC_PATH) -o $(BIN_PATH)/cal3DIO.o

$(BIN_PATH)/cal3DRun.o: $(SRC_PATH)/cal3DRun.c
	$(CC) -c $(SRC_PATH)/cal3DRun.c $(CFLAGS) -I$(INC_PATH) -o $(BIN_PATH)/cal3DRun.o

$(BIN_PATH)/cal3DUnsafe.o: $(SRC_PATH)/cal3DUnsafe.c 
	$(CC) -c $(SRC_PATH)/cal3DUnsafe.c $(CFLAGS) -I$(INC_PATH) -o $(BIN_PATH)/cal3DUnsafe.o



clean:
	@$(RM) $(LIB_PATH)/opencal.a $(BIN_PATH)/*

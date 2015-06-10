include ../../../Makefile.inc

CPP						= g++
INC_PATH 				= ./include
SRC_PATH				= ./source
BIN_PATH				= ./bin
OPENCAL_LIB_PATH		= ../../../OpenCAL/lib
OPENCAL_INC_PATH		= ../../../OpenCAL/include

all: dir $(BIN_PATH)/sciara

dir:
	mkdir -p ./bin 

$(BIN_PATH)/sciara: $(BIN_PATH)/main.o $(BIN_PATH)/Sciara.o $(BIN_PATH)/GISInfo.o $(BIN_PATH)/configurationPathLib.o $(BIN_PATH)/vent.o $(BIN_PATH)/io.o
	$(CPP) -o $(BIN_PATH)/sciara $(BIN_PATH)/main.o $(BIN_PATH)/Sciara.o $(BIN_PATH)/GISInfo.o  $(BIN_PATH)/configurationPathLib.o $(BIN_PATH)/vent.o $(BIN_PATH)/io.o $(OPENCAL_LIB_PATH)/opencal.a
	ln -fs $(BIN_PATH)/sciara .

$(BIN_PATH)/main.o: $(SRC_PATH)/main.cpp
	$(CPP) -c $(SRC_PATH)/main.cpp $(CFLAGS) -I$(OPENCAL_INC_PATH) -I$(INC_PATH) -o $(BIN_PATH)/main.o

$(BIN_PATH)/Sciara.o: $(SRC_PATH)/Sciara.cpp
	$(CPP) -c $(SRC_PATH)/Sciara.cpp $(CFLAGS) -I$(OPENCAL_INC_PATH) -I$(INC_PATH) -o $(BIN_PATH)/Sciara.o

$(BIN_PATH)/GISInfo.o: $(SRC_PATH)/GISInfo.cpp
	$(CPP) -c $(SRC_PATH)/GISInfo.cpp $(CFLAGS) -I$(OPENCAL_INC_PATH) -I$(INC_PATH) -o $(BIN_PATH)/GISInfo.o

$(BIN_PATH)/configurationPathLib.o: $(SRC_PATH)/configurationPathLib.cpp
	$(CPP) -c $(SRC_PATH)/configurationPathLib.cpp $(CFLAGS) -I$(OPENCAL_INC_PATH) -I$(INC_PATH) -o $(BIN_PATH)/configurationPathLib.o

$(BIN_PATH)/vent.o: $(SRC_PATH)/vent.cpp
	$(CPP) -c $(SRC_PATH)/vent.cpp $(CFLAGS) -I$(OPENCAL_INC_PATH) -I$(INC_PATH) -o $(BIN_PATH)/vent.o

$(BIN_PATH)/io.o: $(SRC_PATH)/io.cpp
	$(CPP) -c $(SRC_PATH)/io.cpp $(CFLAGS) -I$(OPENCAL_INC_PATH) -I$(INC_PATH) -o $(BIN_PATH)/io.o

clean:
	@$(RM) sciara $(BIN_PATH)/*
	rm -fr ./bin
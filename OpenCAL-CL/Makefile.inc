CC = gcc
OPENCL_INC_PATH	= /usr/local/cuda/include
OPENCL_UTILS_PATH = ./OpenCL-Utils
OPENCL_UTILS_LIB_PATH	= $(OPENCL_UTILS_PATH)/lib
OPENCL_UTILS_INC_PATH	= $(OPENCL_UTILS_PATH)/include
OPENCAL_INC_PATH	= ../OpenCAL/include
OPENCAL_LIB_PATH	= ../OpenCAL/lib
CFLAGS = -O0 -march=native -pipe -g -Wno-write-strings
#CFLAGS = -g -fopenmp 
#-DPROF

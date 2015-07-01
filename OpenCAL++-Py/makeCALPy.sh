#!/bin/bash

echo "Generating swig files..."
swig  -c++ -python -I./include/swig_interfaces -I./include/ -I../OpenCAL/include -outdir ./lib ./opencal.i
echo "Compiling Python Wrapper..."
g++ -shared -fPIC -fopenmp  opencal_wrap.cxx  ./source/*.cpp ../OpenCAL/source/* -I./ -I../OpenCAL/include/ -I./include/ -I/usr/include/python2.7/  -o lib/_opencal.so
echo "END - library and python wrapper in lib folder"


#in case of problem with omp link it directly /usr/lib .... libgomp

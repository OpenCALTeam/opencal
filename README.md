
OpenCAL - The Cellular Automata Library    [![Build Status](https://travis-ci.org/OpenCALTeam/opencal.svg?branch=master)](https://travis-ci.org/OpenCALTeam/opencal)


Developers should read the [DEVELOPER_README.md](DEVELOPER_README) file and be sure to have fully absorbed the CODE CONVENCTION before to push any code.
** Documentation is important **

OpenCAL README file...

Compiling:
```
mkdir build
cd build
cmake ..
make
```
Example compilation can be controlled using  the following argument to cmake:
-DEXAMPLES:STRING= that takes two possible values: ON or OFF

to change compiler use the following cmake variables: 
```
-DCMAKE_C_COMPILER=
-DCMAKE_CXX_COMPILER=
```
For example, order to compile using clang compiler 
```cmake  -DCMAKE_C_COMPILER=/usr/bin/clang -DCMAKE_CXX_COMPILER=/usr/bin/clang++ -DEXAMPLES:STRING=ON ... ```

*clang does not fully support OpenMP*


OpenCL users. One of the following environment variable have to be  defined.

For example in order to compile with CUDA OpenCL implementation define:

```
export CUDA_PATH="root CUDA FOLDER"
```
(default: /usr/local/cuda/) 
```bash
ENV "PROGRAMFILES(X86)"
ENV AMDAPPSDKROOT
ENV INTELOCLSDKROOT
ENV NVSDKCOMPUTE_ROOT
ENV CUDA_PATH
ENV ATISTREAMSDKROOT
```
To disable OpenCL (and not compile the corrensponding version of OpenCAL) use the following option (defaulted to YES)
```
-DBUILD_DOCUMENTATION:STRING=OFF 
```

Documentation build may be enabled using the cmake option ```-DBUILD_DOCUMENTATION:STRING=ON``` (DOxygen and graphviz are required)



***developers only***
in order to generate an eclipse makefile project run cmake using (x.y are major and minor version of eclipse. Use 4.3 for instance)
```
-G "Eclipse CDT4 - Unix Makefiles" -DCMAKE_ECLIPSE_VERSION:STRING=x.y
```
For example this:
```
cmake -G "Eclipse CDT4 - Unix Makefiles" -DEXAMPLES:STRING=ON -DBUILD_DOCUMENTATION=OFF ..
```
generates an eclipse project into the eclipseproject folder. Import it using the eclipse import menu.

In oder to for cmake to generate dependencies list (internal and to external libraries) use

cmake --graphviz=deps.dot .. && dot -Tps peds.dot -o deps.ps

for example in order to generate deps.ps postscript image that shows dependencies (***inside build directory***)

```
rm -rf * && cmake --graphviz=deps.dot ..  -DEXAMPLES:STRING=ON -DBUILD_DOCUMENTATION=OFF -DBUILD_OPENCL:STRING=ON -DBUILD_GL:STRING=ON -DBUILD_OMP:STRING=ON .. && dot -Tps deps.dot -o deps.ps
```

The cmake option ENABLE_SHARED is used to switch from static and shared object output for the library.



***TESTING THE LIBRARY***
The library comes with a series of performance and correctness tests based on the running of various models with the different implementation of the library. ALl the test files are located in the ***opencal/test*** folder. In order to run the tests do the following.

1. Configure using the options: -DBUILD_OPENCAL_TESTS (example: cd build && rm -rf * && cmake -DBUILD_EXAMPLES:STRING=ON -DBUILD_OPENCAL_PP=ON -DBUILD_OPENCAL_SERIAL=ON -DBUILD_OPENCAL_TESTS:STRING=ON -DBUILD_OPENCAL_OMP:STRING=ON )
2. make
3. cd ../tests && bash runTests.sh

#fixme
In order to add a new tests one should first create a new folder into  opencal/tests. Into this folder then create a number of directories each containing a particular implementation of the test. Foir instance in the life2D test there is an implementation for the serial version of the test (that is the reference output), and an omp implementation plus all the cmake files needed to compile the tests. Tests executable should output the result of the model run into the testsout folder using the convention that filenames are just incremental number. 

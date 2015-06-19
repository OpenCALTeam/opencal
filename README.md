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

Documentation build may be enabled using the cmake option ```-DBUILD_DOCUMENTATION=ON```


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

#OpenCAL - The Open Computing Abstraction Layer for structured grid meshes [![Build Status](https://travis-ci.org/OpenCALTeam/opencal.svg?branch=master)](https://travis-ci.org/OpenCALTeam/opencal)

**OpenCAL** is an Open Source, multi-platform parallel software library for performing fast and reliable simulations of numerical models based on structured grids computational methods, such as the Cellular Automata, Finite Volumes, and others. It is beased on the Extended Cellular Automata (XCA) general formal paradigm.

**OpenCAL** is written in C/C++ and supports Linux/Unix out of the box. It also supports Microsoft Windows through MinGW. Microsoft Visual Studio is currently partially supported.

<img src="https://github.com/OpenCALTeam/OpenCALTeam.github.io/blob/master/assets/timer_icon.png" width="48">
Gives you the power to concentrate only on simulation code. No memory or parallelism management required.

<img src="https://github.com/OpenCALTeam/OpenCALTeam.github.io/blob/master/assets/rocket_icon.png" width="48">
Fast execution on multiple platforms. It exploits multicore CPUs, many-core GPUs, as well as clusters of workstations.

<img src="https://github.com/OpenCALTeam/OpenCALTeam.github.io/blob/master/assets/docs_icon.png" width="48">
Code documented and mantained.

<img src="https://github.com/OpenCALTeam/OpenCALTeam.github.io/blob/master/assets/opensource_icon.png" width="48">
Open source project released under the LGPLv3 license.


***


<!-- Developers should read the [DEVELOPER_README.md](DEVELOPER_README.md) file and be sure to have fully absorbed the CODE CONVENCTION before to push any code.-->


#Requirements and dependencies

<ul>
	<li> CMake 2.8 (CMake 3.1 is needed to complile OpenCAL-CL).
	<li> A quite recent C/C++ compiler (Full support to OpenMP 4 is needed to compile OpenCAL-OMP).
	<li> OpenGL/GLUT is also needed to compile OpenCAL-GL, which provides a minimal User Interface and visualization system to OpenCAL applications.
	<li> Doxygen and Graphviz to build documentation.
</ul>


#Making and installing


```
user@machine:~/git/opencal-1.0$ mkdir build
user@machine:~/git/opencal-1.0$ cd build
user@machine:~/git/opencal-1.0/build$ cmake ../ [-DBUILD_OPENCAL_SERIAL=ON|OFF] [-DBUILD_OPENCAL_OMP=ON|OFF] [-DBUILD_OPENCAL_CL=OFF|ON] [-DBUILD_OPENCAL_GL=OFF|ON] [-DBUILD_DOCUMENTATION=OFF|ON] [-DCMAKE_INSTALL_PREFIX:PATH=/usr/local | custom_install_path]
user@machine:~/git/opencal-1.0/build$ make | tee make.log
user@machine:~/git/opencal-1.0/build$ make install | tee install.log
```

Arguments in square brackets are optional. Default value is shown first, other possible values are separated by pipes.

<!--
Example compilation can be controlled using  the following argument to cmake:
-DEXAMPLES:STRING= that takes two possible values: ON or OFF
-->
To change compiler use the following CMake variables:

```
-DCMAKE_C_COMPILER=
-DCMAKE_CXX_COMPILER=
```

For example, order to compile using clang

```
cmake  -DCMAKE_C_COMPILER=/usr/bin/clang -DCMAKE_CXX_COMPILER=/usr/bin/clang++ -DEXAMPLES:STRING=ON ... ```
```

***Developers only***
In order to generate an eclipse makefile project run CMake using (x.y are major and minor version of eclipse. Use 4.3 for instance)
```
-G "Eclipse CDT4 - Unix Makefiles" -DCMAKE_ECLIPSE_VERSION=x.y
```
For example this:
```
cmake -G "Eclipse CDT4 - Unix Makefiles" -DEXAMPLES=ON -DBUILD_DOCUMENTATION=OFF ..
```
generates an eclipse project into the eclipse project folder. Import it using the eclipse import menu.

In oder to generate dependencies list (internal and to external libraries) use

cmake --graphviz=deps.dot .. && dot -Tps peds.dot -o deps.ps

For example in order to generate deps.ps postscript image that shows dependencies (***inside build directory***)

```
rm -rf * && cmake --graphviz=deps.dot ..  -DEXAMPLES:STRING=ON -DBUILD_DOCUMENTATION=OFF -DBUILD_OPENCL:STRING=ON -DBUILD_GL:STRING=ON -DBUILD_OMP:STRING=ON .. && dot -Tps deps.dot -o deps.ps
```

The CMake option ENABLE_SHARED is used to switch from static and shared object output for the library. Default value is ON.



***TESTING THE LIBRARY***
The library comes with a series of performance and correctness tests based on the running of various models with the different implementation of the library. All the test files are located in the ***test*** directory. In order to run the tests do the following.

1. Configure using the options: -DBUILD_OPENCAL_TESTS (example: cd build && rm -rf * && cmake -DBUILD_EXAMPLES=ON -DBUILD_OPENCAL_PP=ON -DBUILD_OPENCAL_SERIAL=ON -DBUILD_OPENCAL_TESTS=ON -DBUILD_OPENCAL_OMP=ON )
2. make
3. cd ../tests && bash runTests.sh

#fixme
In order to add a new tests one should first create a new folder into  opencal/tests. Into this folder then create a number of directories each containing a particular implementation of the test. For instance, in the life2D test there is an implementation for the serial version of the test (that is the reference output), and an OpenMP implementation plus all the CMake files needed to compile the tests. Tests executable should output the result of the model run into the testsout folder using the convention that filenames are just incremental number.

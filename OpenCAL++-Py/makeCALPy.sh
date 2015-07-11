#!/bin/bash

#FUNCTION ---------------------------------------------
ask() {
    while true; do
 
        if [ "${2:-}" = "Y" ]; then
            prompt="Y/n"
            default=Y
        elif [ "${2:-}" = "N" ]; then
            prompt="y/N"
            default=N
        else
            prompt="y/n"
            default=
        fi
 
        # Ask the question - use /dev/tty in case stdin is redirected from somewhere else
        read -p "$1 [$prompt] " REPLY </dev/tty
 
        # Default?
        if [ -z "$REPLY" ]; then
            REPLY=$default
        fi
 
        # Check if the reply is valid
        case "$REPLY" in
            Y*|y*) return 0 ;;
            N*|n*) return 1 ;;
        esac
 
    done
}
#------------------------------------------------------

#usage: path to be added
#variable to append to
pathadd() {
eval "VAR=\$$2";
echo "$2 = $VAR"
	if [[ $VAR = "" ]]; then
   		export $2="$1"
	fi

    if [[ ! $VAR =~ (^|:)$1(:|$) ]]; then
		export $2="$VAR:$1"
    fi
	
}

CC_DEBUG="-g3 -ggdb"
echo "Generating swig files..."
swig  -c++ -python -builtin -I./include/swig_interfaces -I../OpenCAL++/include -outdir ./lib ./opencal.i
echo "Compiling Python Wrapper..."
g++ -O2 -shared  -fPIC -fopenmp  opencal_wrap.cxx  ../OpenCAL++/source/* -I./ -I../OpenCAL++/include/  -I/usr/include/python2.7/  -o lib/_opencal.so

# exporting ldlibrarypath of opencal.so:
if ask "Do you want to export _opencal.so in LD_LIBRARY_PATH? (make sure that you are in the root of OpenCAL++-Py and that the script is called using *source*) and pdate the PYTHONPATH" Y; then
    echo "YES";
	pathadd "`pwd`/lib" LD_LIBRARY_PATH
	pathadd "`pwd`/lib" PYTHONPATH 
 
    if ask "Do you want to add the exports to .bashrc file?" N; then
      echo "YES";
      echo "export PYTHONPATH=$PYTHONPATH" >> ~/.bashrc;
      echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >> ~/.bashrc;
    fi

else
    echo "No"
fi




echo "END - library and python wrapper in lib folder."

#in case of problem with omp link it directly /usr/lib .... libgomp





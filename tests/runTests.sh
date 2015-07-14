#!/bin/bash

#exit if some variables are not initialized
set -o nounset
#exit on error
set -o errexit

#functions --------------------------
BLACK="\e[30m"
RED="\e[31m"
GREEN="\e[32m"
YELLOW="\e[33m"
BLUE="\e[34m"
WHITE="\e[37m"
printColored() {
	echo -e "$1$2$WHITE"
}

Indent() {
	 echo "$1" | sed -e 's/^/    /'
}

ExecutableExistsOrDie(){
bin="$1"
#check if the executable exists (opencal not built? was BUILD_OPENCAL_TESTS cmake option used?
if [ ! -e "$bin" ] ; then
	printColored $WHITE "FATAL ERROR- Executable $bin does not exists"	
	Indent "$(printColored $WHITE "Was opencal built?")"
	Indent "$(printColored $WHITE "Did you use BUILD_OPENCAL_TESTS during configuration?")"
	Indent "$(printColored $WHITE "EXITING . . .")"
	exit 1 
fi		
}
#functions --------------------------

#SCRIPT STARTS

#tests should always be launched from opencal/tests directory!
TESTROOT=`pwd`
ISCALTESTDIR=$(basename $(dirname $TESTROOT))/$(basename $TESTROOT)
if [[ $ISCALTESTDIR != "opencal/tests" ]]; then
	printColored $WHITE "Please launch tests from opencal/test directory"	
	printColored $WHITE "EXITING . . ."
	exit 2
	
fi

mkdir -p testsout/serial
mkdir -p testsout/other
for d in */ ; do
	if [[ $d != "testsout/" ]]; then
		dir=${d%/}
		printColored $GREEN "TEST SUITE $dir";
		#execute serial version
		bin=$dir/cal-$dir/bin/cal-$dir-test

		ExecutableExistsOrDie "$bin"

		Indent "$(printColored $BLUE "Creating Reference Test Data (Serial Version)  $bin 0")"
		./$bin 0

		#now run all the otherversion of this test
		cd $dir
		for o in */ ; do
			if [[ $o != "cal-$dir/" ]]; then
				odir=${o%/}
				#execute serial version
				otherBin=$dir/$odir/bin/$odir-test
								
				#need to run the test from the tests directory!					
				cd ..
				ExecutableExistsOrDie "$otherBin"

				Indent "$(printColored $RED "Executing  $odir-test")"
				#execute test
				./$otherBin 1
				#restore test directory to run other version of the test
				cd $dir

				#NOW RUN THE SCRIPT CHECK

				#md5sum CUMULATIVE
				MD5ALLSERIALS="$(cat $TESTROOT/testsout/serial/*   | md5sum)"
				MD5ALLOTHERS="$(cat $TESTROOT/testsout/other/*	   | md5sum)"
				RES="OK"
				if [[ "$MD5ALLSERIALS" != "$MD5ALLOTHERS" ]]; then
					Indent "$(Indent "$(printColored $YELLOW "MD5 CUMULATIVE FAILED. VERSION=$o")")"	
					exit 2 
				else
					Indent "$(Indent "$(printColored $GREEN "MD5 CUMULATIVE OK. VERSION=$o")")"
				fi

				
				
				
			fi
		done
		
	fi
done
#cd .. && rm testsout/serial/*.txt testsout/other/*.txt
printColored $WHITE "END";





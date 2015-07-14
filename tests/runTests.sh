#!/bin/bash

BLACK="\e[30m"
RED="\e[31m"
GREEN="\e[32m"
YELLOW="\e[33m"
BLUE="\e[34m"
printColored() {
	echo -e "$1$2$BLACK"
}

Indent() {
	 echo "$1" | sed -e 's/^/    /'
}

for d in */ ; do
	if [[ $d != "testsout/" ]]; then
		dir=${d%/}
	    printColored $GREEN "TEST SUITE $dir";
		#execute serial version
		bin=$dir/cal-$dir/bin/cal-$dir-test
	    Indent "$(printColored $BLUE "Creating Reference Test Data (Serial Version)  $bin 0")"
		./$bin 0

		#now run all the otherversion of this test
		cd $dir
		for o in */ ; do
			if [[ $o != "cal-$dir/" ]]; then
				odir=${o%/}
				#execute serial version
				otherBin=$dir/$odir/bin/$odir-test
			    Indent "$(printColored $RED "Executing  $odir-test")"
				#need to run the test from the tests directory!	
				cd ..	&& 		./$otherBin 1 	&& cd $dir
				
				
			fi
		done
		
	fi
done
echo -e "\e[39mPASSED"





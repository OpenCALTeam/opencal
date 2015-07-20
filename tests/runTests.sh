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
PURPLE="\e[35m"
DEFAULT="\e[39m"

NUMERICALTEST=1
MD5TEST=0

EPSILON=0.0001
printColored() {
	echo -e "$1$2$DEFAULT"
}

Indent() {
	 echo "$1" | sed -e 's/^/    /'
}

Exiting() {
	Indent "$(printColored $RED "EXITING . . .")"
	exit 1 
}

ExecutableExistsOrDie(){
bin="$1"
#check if the executable exists (opencal not built? was BUILD_OPENCAL_TESTS cmake option used?
if [ ! -e "$bin" ] ; then
	printColored $RED "FATAL ERROR- Executable $bin does not exists"	
	Indent "$(printColored $RED "Was opencal built?")"
	Indent "$(printColored $RED "Did you use BUILD_OPENCAL_TESTS during configuration?")"
	Exiting
fi		
}

Md5CumulativeTest() {
	#md5sum CUMULATIVE
	MD5ALLSERIALS="$(cat $TESTROOT/testsout/serial/*   | md5sum)"
	MD5ALLOTHERS="$(cat $TESTROOT/testsout/other/*	   | md5sum)"
	RES="OK"
	if [[ "$MD5ALLSERIALS" != "$MD5ALLOTHERS" ]]; then
		Indent "$(Indent "$(printColored $RED "MD5 CUMULATIVE FAILED.")")"	
		#Exiting
	else
		Indent "$(Indent "$(printColored $GREEN "MD5 CUMULATIVE OK.")")"
	fi

}

Md5CompareTwoFile() {
refFile="$1"
otherFile="$2"
base=`basename $refFile`
 	
		#if it exists performs the md5sum check on it
		#(use awk to take only the md5hex output) md5sum output: HEX <space> filename
		MD5REF=`md5sum 	$refFile 	| awk '{print $1;}'`
		MD5OTH=`md5sum 	$otherFile  | awk '{print $1;}'`
		if [[ "$MD5REF" != "$MD5OTH" ]]; then
			Indent "$(Indent "$(printColored $RED "MD5 FAILED: $base ")")"
			Indent "$(Indent "$(Indent "$(printColored $RED "$MD5REF : $MD5OTH")")")"		
						#Exiting
		else
		#md5sum on this file OK								
			Indent "$(Indent "$(printColored $GREEN "MD5: $base OK")")"
			#Indent "$(Indent "$(Indent "$(printColored $GREEN "$MD5REF : $MD5OTH")")")"
		fi

}


CompareMatrixEpsilon() {
	refFile="$1"
	othFile="$2"
	base=`basename $refFile`
	OUT="$(perl $TESTROOT/compareMatrix.pl $refFile $othFile $EPSILON" ")"
	if [[ "$OUT" == "OK" ]]; then
		Indent "$(Indent "$(printColored $GREEN "NUMERICAL COMPARISON(e=$EPSILON): $base $OUT")")"
	else
		Indent "$(Indent "$(printColored $RED "NUMERICAL COMPARISON(e=$EPSILON): $base $OUT")")"
	fi
}

#functions --------------------------
TestOutputFiles() {
	#md5sum of one of the output file at time. Usefule if not all the output is wrong
	testType=$1
	for refFile in "$TESTROOT/testsout/serial"/*
	do
	base=`basename $refFile`
	otherFile="$TESTROOT/testsout/other/"$base
	
	  if [ -f "$refFile" ];then
		#check that the same file exists in the 'other' directory
			if [ -f "$otherFile" ];then
				#CHOOSE THE APPROPRIATE TEST
				case "$testType" in
					"$MD5TEST")
						Md5CompareTwoFile "$refFile" "$otherFile"
						;;
					"$NUMERICALTEST")
						CompareMatrixEpsilon "$refFile" "$otherFile"
						;;
					*)
					 printColored $RED "Invalid test type"
					Exiting
						;;
				esac		
			#in case the other file does not exists
			else
				Indent "$(Indent "$(printColored $RED "OUTPUT $base NOT FOUND")")"	
				Exiting
	  		fi	
	  fi
	done

	RES="OK"
	

}

ExecuteAndSaveElapsedTime() {
	outFile="$1"
	binary="$2"
	parameters="$3"
	execTime="$(time ( ./$binary $parameters) 2>&1 1>/dev/null )"
	Indent "$(printColored $PURPLE "Elapsed Time: $execTime")"
	echo -e "\tTEST $binary" >> $TIMINGFILE;	
	echo -e "\t $execTime\n" >> $outFile
}


#functions --------------------------


#------------SCRIPT STARTS------------#
#numerical accuracy for the tests


#tests should always be launched from opencal/tests directory!
TESTROOT=`pwd`
ISCALTESTDIR=$(basename $(dirname $TESTROOT))/$(basename $TESTROOT)

if [[ $ISCALTESTDIR != "opencal/tests" ]]; then
	printColored $DEFAULT "Please launch tests from opencal/test directory"	
	printColored $DEFAULT "EXITING . . ."
	exit 2
	
fi

mkdir -p testsout/serial
mkdir -p testsout/other
rm -f testsout/serial/*
rm -f testsout/other/*
TIMINGFILE="TestTiming-`date +"%s"`"
touch $TIMINGFILE
for d in */ ; do
	if [[ $d != "testsout/" &&  $d != "testData/" ]]; then
		dir=${d%/}
		
		echo ""
		printColored $GREEN "TEST SUITE $dir";
		echo "TEST SUITE $dir" >> $TIMINGFILE;
		#printColored $GREEN "`cat $dir/description.txt`"; #uncomment if you want to print a description of the test
		

		#execute serial version
		bin=$dir/cal-$dir/bin/cal-$dir-test

		
		ExecutableExistsOrDie "$bin"

		Indent "$(printColored $BLUE "Creating Reference Test Data (Serial Version)  $bin 0")"
		#./$bin 0 
		ExecuteAndSaveElapsedTime $TIMINGFILE $bin 0

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

				Indent "$(printColored $YELLOW "Executing  $odir-test")"
			#execute test
				 #./$otherBin 1

				ExecuteAndSaveElapsedTime $TIMINGFILE $otherBin 1

			#md5sum on all single files
				TestOutputFiles $MD5TEST
			#md5sum CUMULATIVE
				Md5CumulativeTest
			#Numerical comparison
				TestOutputFiles $NUMERICALTEST
				
				rm -f $TESTROOT/testsout/serial/*
				rm -f $TESTROOT/testsout/other/*
				
		#restore test directory to run other version of the test
				cd $TESTROOT
				
				
			fi
		done
		
	fi
done
rm -f $TESTROOT/testsout/serial/*
rm -f $TESTROOT/testsout/other/*
#cd .. && rm testsout/serial/*.txt testsout/other/*.txt
printColored $DEFAULT "END";

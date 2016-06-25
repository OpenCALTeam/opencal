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
NUMBEROFTESTS=1
STEPS=2
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

function pause(){
   read -p "$*"
}

ExecutableExistsOrDie(){

binary="$1"
#check if the executable exists (opencal not built? was BUILD_OPENCAL_TESTS cmake option used?
file $binary
if [ ! -f "$binary" ] ; then
	printColored $RED "FATAL ERROR- Executable $binary does not exists"
	printColored $RED $OUT
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
#	timeUtility="/usr/bin/time"
#	format="Time\t%E\tMaxMem\t%M\tPageFaults(minor)\t%R\tPageFaults(major)\t%F\tFS(input)\t%I\tFS(outpu)\t%O"
# format=" -v "
#	timeOptions=" -f $format "
	outFile="$1"
	binary="$2"
	parameters="$3"


	#echo "here $timeUtility $timeOptions ./$binary $parameters 2>&1"
#	execTime="$( $timeUtility $timeOptions ./$binary $parameters 2>&1)"


        if [[ $binary == *"calcl"* ]]
        then
            #echo "ENTRO IF calcl"
            #Get all platforms and devices
            #For each device executes the examples for NUMBEROFTESTS times and save the results in milliseconds
            echo -e "\tTEST ON $binary" >> $TIMINGFILE;
            getPlatformAndDevice="$(./getOpenCLPlatformAndDevice/bin/calcl-getPlatformAndDevice-test 2>&1)"
            allPlatformAndDevice=(${getPlatformAndDevice// / })
            for i in "${allPlatformAndDevice[@]}"
            do
                #echo $i
                platformAndDevice=(${i//;/ })
                echo "Execution on Plarform ${platformAndDevice[0]} and Device ${platformAndDevice[1]}"
                totalmilliseconds=0
                for (( c=1; c<= $NUMBEROFTESTS; c++ ))
                do
                    execTime="$(./$binary $parameters $STEPS ${platformAndDevice[0]} ${platformAndDevice[1]} 2>&1)"
                    #split outpute, take only milliseconds
                    times=(${execTime//;/ })
                    #times=$(echo $execTime | tr ";" "\n")
                    echo "Execution simulation number $c "

                    let "totalmilliseconds += ${times[1]}"
                 done

                let "tmp = $totalmilliseconds / $NUMBEROFTESTS"

                Indent "$(printColored $PURPLE "Elapsed Time: $tmp")"
                echo -e "\tTEST ON ${platformAndDevice[0]} and Device ${platformAndDevice[1]}" >> $TIMINGFILE;
                echo -e "\t $tmp \n" >> $outFile
            done

        else
            #echo "ENTRO ELSE calcl"
            totalmilliseconds=0
            for (( c=1; c<= $NUMBEROFTESTS; c++ ))
            do
                 execTime="$(./$binary $parameters $STEPS 2>&1)"
                 #split outpute, take only milliseconds
                 times=(${execTime//;/ })
                 #times=$(echo $execTime | tr ";" "\n")
                 echo "Execution simulation number $c "

                let "totalmilliseconds += ${times[1]}"
                #echo $totalmilliseconds
                #totalmilliseconds += ${times[1]}
            done

            let "tmp = $totalmilliseconds / $NUMBEROFTESTS"

            Indent "$(printColored $PURPLE "Elapsed Time: $tmp")"
            echo -e "\tTEST $binary" >> $TIMINGFILE;
            echo -e "\t $tmp \n" >> $outFile

        fi


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
TIMINGFILE="TestTiming-`date +"%d-%B-%y_%R:%S"`"
touch $TIMINGFILE
for d in */ ; do
#        if [[ $d != "include/" && $d != "testsout/" &&  $d != "testData/" &&  $d != "plotFiles/" ]]; then
#        if [[ $d == "sciddicaT/" || $d == "heattransfer/" ]]; then
        if [[ $d == "sciddicaT/" ]]; then
		dir=${d%/}

                echo ""
                printColored $GREEN "TEST SUITE $dir";
                echo "SUITE $dir" >> $TIMINGFILE;
#		#printColored $GREEN "`cat $dir/description.txt`"; #uncomment if you want to print a description of the test


#		#execute serial version
                bin=$dir/cal-$dir/bin/cal-$dir-test


#		ExecutableExistsOrDie "$bin"

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
                                echo "TESTING $otherBin"
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
#comment this line to avoid pause between tests
#pause
                                rm -f $TESTROOT/testsout/other/*

                #restore test directory to run other version of the test
                                cd $dir


                        fi
                done

                rm -f $TESTROOT/testsout/serial/*
                cd $TESTROOT
	fi
done
rm -f $TESTROOT/testsout/serial/*
rm -f $TESTROOT/testsout/other/*
#cd .. && rm testsout/serial/*.txt testsout/other/*.txt
printColored $DEFAULT "END";

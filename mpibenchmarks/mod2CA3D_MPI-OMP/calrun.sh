#!/bin/bash

###############################################################################
#load utils functions
#source opencalTestUtils.sh
###############################################################################
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
NUMBEROFLOOPS=200
STEPS=4000
MD5TEST=0

EPSILON=0.0001
printColored() {
	COLOR="$1"
	shift
	REST="$*"
	echo -e "$COLOR$REST$DEFAULT"
	
}

Indent() {
	 echo "$*" | sed -e 's/^/    /'
}

Exiting() {
	Indent "$(printColored $RED "EXITING . . .")"
	exit 1
}

function pause(){
   read -p "$*"
}

GetIpfromClusterFile(){
	FILE="$1"
	IPS=$(grep -o '[0-9]\{1,3\}\.[0-9]\{1,3\}\.[0-9]\{1,3\}\.[0-9]\{1,3\}' "$FILE")
	echo "$IPS"
}
###############################################################################

ParseAndRun(){
  MPI_JOB="$1"
  FILE="$2";
  HOSTFILE="$3";
  #remove first three arguments
  shift
  shift
  shift
  MPI_OPTS="$*";
  if [ ! -f "$FILE" ] ; then
    printColored $RED "FATAL ERROR- Cluster definition $FILE does not exists"
    printColored $RED $OUT
    Exiting
  fi
  MPI_NP=""
  MPI_MACHINEFLAG=""
  MPICH="0"
  OPENMPI="0"
  if mpiexec --version | grep -q 'mpich'; then
    echo "find mpich"
    MPI_NP="-n"
    MPI_MACHINEFLAG="-f"
    MPICH="1"
  fi

  if mpiexec --version | grep -q 'openmpi'; then
    echo "find openmpi"
    MPI_NP="-np"
    MPI_MACHINEFLAG="--machinefile"
    OPENMPI="1"
  fi 

  if mpirun --version | grep -q 'Open MPI'; then
    echo "find openmpi"
    MPI_NP="-np"
    MPI_MACHINEFLAG="--machinefile"
    OPENMPI="1"
  fi 

  if $OPENMPI="0" && $MPICH="0";
  then
    printColored $RED "FATAL ERROR: MPI not found."
    exit
  fi

  IPS=$(grep -o '[0-9]\{1,3\}\.[0-9]\{1,3\}\.[0-9]\{1,3\}\.[0-9]\{1,3\}' "$FILE")
  echo "BEGIN IPS" 
  echo "$IPS"
  echo "END IPS"
  #create a temporary file IPSFile in order to sort all the ips with unique values  
  $(grep -o '[0-9]\{1,3\}\.[0-9]\{1,3\}\.[0-9]\{1,3\}\.[0-9]\{1,3\}' "$FILE" > IPSFile)
  #number od repetition for each IPS
  NRAIP=$(sort "IPSFile" | uniq -c)
  IPSUNIQUE=$(sort -u "IPSFile")

  echo "BEGIN IPSUNIQUE" 
  echo "$NRAIP"
  echo "END IPSUNIQUE" 

  echo "BEGIN IPSUNIQUE" 
  echo "$IPSUNIQUE"
  echo "END IPSUNIQUE" 

  NIPS=`echo $IPS | wc -w`
  NIPSUNIQUE=`echo $IPSUNIQUE | wc -w`
  echo "NIPSUNIQUE = $NIPSUNIQUE"

  TMP=($NRAIP)

  echo "BEGIN NRAIP" 

  echo "END NRAIP" 


  echo "NIPS = $NIPS"
  NNODES=$(sed -n '2p' $FILE)
  if [[ "$NIPS" != "$NNODES" ]]; then
    printColored $RED "NUMBER OF IPs and the specified number of nodes don't match."
    Exiting
  else
    #remove any existent generated hostfile
    rm -f $HOSTFILE
    #generate hostfile

    NC=0;

    for ((i=0;i < $NIPSUNIQUE;i++));
    do
      tone=$NC

      let "NC=NC+1"
      ttwo=$NC 
      #echo " $tone $ttwo "
      if [ "$MPICH" = "1" ]; then
        echo "${TMP[$ttwo]}:${TMP[$tone]}">> $HOSTFILE	
      fi

      echo "OPENMPI = $OPENMPI"
      if [ "$OPENMPI" = "1" ]; then
        echo "${TMP[$ttwo]} slots=${TMP[$tone]}">> $HOSTFILE
      fi
      let "NC=NC+1"
    done

    #for IP in $IPS; do
    #	echo "$IP slots=1"  >> $HOSTFILE
    #done
    #run MPI job
    printColored $GREEN "Executing MPI job $MPI_JOB on nodes with following IPs:"

    Indent $(printColored $GREEN "$IPS")
    printColored $PURPLE "Using the following MPI job invocation: "
    INVOCATION_STR="mpiexec $MPI_NP $NIPS $MPI_MACHINEFLAG $HOSTFILE  $MPI_OPTS ./$MPI_JOB $FILE"		
    Indent $(printColored $RED "$INVOCATION_STR")
    printColored $DEFAULT "RUNNING..."

    mpiexec $MPI_NP $NIPS $MPI_MACHINEFLAG $HOSTFILE  $MPI_OPTS ./$MPI_JOB $FILE

      fi

    rm -f IPSFile $HOSTFILE 

    }



  PrintUsage(){
    printColored $RED "--USAGE--"
    printColored $RED "opencalexec.sh CLUSTERFILE MPI_JOB_EXECUTABLE MPI_EXEC_OPTIONS"
    printColored $GREEN "Example:"
    printColored $GREEN "bash opencalexec.sh clusterfile_onenode.conf mytest --prefix /usr/local/lib -x OPENCALCL_PATH_MULTIGPU=/usr/local/opencal-1.1/include/OpenCAL-CL/"
    echo ""
    echo ""

  }
#echo "start"
#PrintUsage
HOSTFILE="generated_hostfile"
FILE="$1"
MPI_JOB="$2"
shift
shift
MPI_OPTS="$*"
#echo $MPI_OPTS
ParseAndRun $MPI_JOB $FILE $HOSTFILE $MPI_OPTS

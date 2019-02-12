#!/bin/bash

#load utils functions
source opencalTestUtils.sh

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
	

	IPS=$(grep -o '[0-9]\{1,3\}\.[0-9]\{1,3\}\.[0-9]\{1,3\}\.[0-9]\{1,3\}' "$FILE")
	
	NIPS=`echo $IPS | wc -w`
	NNODES=$(sed -n '2p' $FILE)
	if [[ "$NIPS" != "$NNODES" ]]; then
		printColored $RED "NUMBER OF IPs and the specified number of nodes don't match."
		Exiting
	else
		#remove any existent generated hostfile
		rm -f $HOSTFILE
		#generate hostfile
		for IP in $IPS; do
			echo "$IP slots=1"  >> $HOSTFILE
		done
		#run MPI job
		printColored $GREEN "Executing MPI job $MPI_JOB on nodes with following IPs:"
		
		Indent $(printColored $GREEN "$IPS")
		printColored $PURPLE "Using the following MPI job invocation: "
		INVOCATION_STR="mpiexec -np $NIPS --machinefile $HOSTFILE  $MPI_OPTS ./$MPI_JOB $FILE"		
		Indent $(printColored $RED "$INVOCATION_STR")
		printColored $DEFAULT "RUNNING..."
		
		
		mpiexec -np $NIPS --machinefile $HOSTFILE  $MPI_OPTS ./$MPI_JOB $FILE
		
	fi
	
	
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


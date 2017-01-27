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


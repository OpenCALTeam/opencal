#!/bin/bash

#exit if some variables are not initialized
set -o nounset
#exit on error
set -o errexit


GetNthWordOfString() {
  echo `echo $1 | cut -d " " -f $2`

}


metric=""
mkdir -p plotFiles
rm -rf plotFiles/*
plotoutFolder="plotout"
for i in `seq 1 6`;
do

namefile=""
m=$i
  if [[ "$m" == "1" ]]; then
    gnufile="plot_time.plg"
  else
    gnufile="plot_normal.plg"
  fi
  while  read -r line || [[ -n "$line" ]]; do
      if [[ "$line" == "SUITE"* ]]; then
        if [[ "$namefile" != "" ]]; then
        gnuplot -e "datafile='${namefile}'; outputname='${namefile##*/}.png'" $gnufile
        fi
        s=0;
        namefile="plotFiles/`GetNthWordOfString "$line" 2`_$m"
        touch $namefile
        echo $namefile
      else

        if [[ "$line" == "TEST"* ]]; then
          read -r row || [[ -n "$row" ]]
          metric=`GetNthWordOfString "$row" $(($m*2 -1 ))`
          #echo $s ${line##*/} `GetNthWordOfString "$row" $(($m*2 )) | tr . :` >> $namefile
          echo $s ${line##*/} `GetNthWordOfString "$row" $(($m*2 ))` >> $namefile
        #  echo $s ${line##*/} `GetNthWordOfString "$row" $(($2*2 ))` $namefile
          s=$(($s+1 ))
        fi
      fi
  done < "$1"
  gnuplot -e "datafile='${namefile}'; outputname='${namefile##*/}.png'" $gnufile
done #for

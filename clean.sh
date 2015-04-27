 #!/bin/bash

ROOT=$HOME/git/opencal
ELMTS=/tmp/toBeCleaned.txt

#remove log and log files, modify accordingly
find $ROOT -name *.log -or -name *.tlog > $ELMTS

while read line           
do           
    rm -rf $line
	echo "removing $line"           
done <$ELMTS 


#remove log and log files, modify accordingly
find $ROOT -type d -name *bin > $ELMTS

while read line           
do           
    rm -rf $line
	echo "Deleting folder: $line"           
done <$ELMTS 


rm $ELMTS 

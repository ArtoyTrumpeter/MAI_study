#!/bin/bash
if [ $# -lt 3 ] ;
	then
	echo "Please use the command like this: lab21 [filename] [times] [byte-limit]"
	exit 1
fi

filename=$1
times=$2
limit=$3

echo "Times to print: $filename"
echo "Times to print: $times"
echo "Max result size in bytes: $limit"

steps=$((limit / $( stat --printf="%s" $filename )))

if [ $steps -lt $times ]
then
	times=$steps
fi

for i in $( seq $times )
	do
		cat $filename
	done

#!/bin/bash 

source ~/DCDB/install/dcdb.bash

while read p; do
	#echo "$p"
	outputname=${p////}
	echo "$outputname"
	output="$outputname.csv"
	LD_PRELOAD=./isatty.so dcdbquery -r -h 127.0.0.1 $p $1 $2 | tail -n +3 >> $output 
done <sensorlist_data.txt
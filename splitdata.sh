#!/bin/bash 

source ~/DCDB/install/dcdb.bash

function query_and_store
{
	LD_PRELOAD=./isatty.so dcdbquery -r -h 127.0.0.1 $1 $2 $3 | tail -n +3 > $4
}

i=0
while read p; do
	i=$((i+1))
	#echo "$p"
	outputname=${p////}
	echo "$outputname"
	output="$outputname.csv"
	#execute it in background in parallel
	query_and_store $p $1 $2 $output &
	#if we have launched $3 tasks, wait for them to finish before starting next batch
	if ! ((i % $3)); then
		wait
	fi
done <sensorlist_data.txt

wait
echo "All CSVs have been retrieved"
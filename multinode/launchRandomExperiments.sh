#!/bin/bash

declare -a savefolders=()

csvpath=$1
savepath=$2
tmin=$3
tmax=$4

for n in {1..5}
do
	savefolders=("${savefolders[@]}" RN"$n"d)
done

declare -a resultsfolders=()

#create result directory if it does not exist
for folder in "${savefolders[@]}"
do
	mkdir -p "$savepath"/"$folder"
	resultsfolders=("${resultsfolders[@]}" "$savepath"/"$folder")
done

#execute the experiments
for i in {0..4}
do
	k=$((i+1))
	echo "Starting experiment $k"
	python -u randomMultinodeTrainAndTest.py -p "$csvpath" -v "${resultsfolders[$i]}" -k "$k" -b "$tmin" -c "$tmax" 2>&1 | tee logfile_1.log
	echo "Finished experiment $k"
done
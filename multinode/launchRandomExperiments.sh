#!/bin/bash

declare -a savefolders=()

csvpath=$1
savepath=$2
sumpath=$3
tmin=$4
tmax=$5

for n in {1..5}
do
	savefolders=("${savefolders[@]}" RN"$n")
done

declare -a resultsfolders=()

#create result directory if it does not exist
for folder in "${savefolders[@]}"
do
	mkdir -p "$savepath"/"$folder"
	resultsfolders=("${resultsfolders[@]}" "$savepath"/"$folder")
done

#create summary folder if it does not exist
mkdir -p "$sumpath"

#execute the experiments
for i in {0..4}
do
	k=$((i+1))
	echo "Starting experiment $k"
	python -u randomMultinodeTrainAndTest.py -p "$csvpath" -v "${resultsfolders[$i]}" -d "$sumpath" -k "$k" -b "$tmin" -c "$tmax" | tee logfile_allfeats.log
	echo "Finished experiment $k"
done
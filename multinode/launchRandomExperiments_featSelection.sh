#!/bin/bash

declare -a savefolders=()

csvpath=$1
featfile=$2
savepath=$3
sumpath=$4
step=$5

for n in {1..16}
do
	savefolders=("${savefolders[@]}" RN"$n"n_f)
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
for i in {0..15}
do
	k=$((i+1))
	echo "Starting experiment $k"
	python -u randomMultinodeTrainAndTest_featureSelected.py -p "$csvpath" -v "${resultsfolders[$i]}" -k "$k" -d "$sumpath" -e "$featfile" -f "$step" 2>&1 | tee logfile_2.log
	echo "Finished experiment $k"
done
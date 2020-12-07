#!/bin/bash

declare -a savefolders=()

csvpath=$1
featfile=$2
savepath=$3
sumpath=$4
step=$5
balancetest_default="yes"
balancetest=${6:-$balancetest_default}
shuffling_default="no"
shuffling=${7:-$shuffling_default}
feattype_default="rfe"
feattype=${8:-$feattype_default}
numfeat_default="100"
numfeat=${9:-$numfeat_default}

for n in {1..16}
do
	savefolders=("${savefolders[@]}" RN"$n"_f)
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
	python -u randomMultinodeTrainAndTest_featureSelected.py -p "$csvpath" -v "${resultsfolders[$i]}" -k "$k" -d "$sumpath" -e "$featfile" -i "$feattype" -l "$numfeat" -f "$step" -g "$balancetest" -s "$shuffling" | tee "logfile_${k}.log"
	echo "Finished experiment $k"
done
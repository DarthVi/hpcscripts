#!/bin/bash

declare -a savefolders=()
savepath="/run/media/vincenzo/HDATA/CovellaScript/multinode/random_multinode_newdataset_experiments/LowerN9_AboveN24_excluded_from_training/10runs_100features_bothunbalanced_noshuffling_nosampling_allmetrics_fixedclfseed/"

for n in {1..5}
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

#execute the experiments
for i in {0..4}
do
	k=$((i+1))
	echo "Starting experiment $k"
	python -u inmemoryMultirunRandomMultinodeTrainAndTest_featureSelected_allmetrics_updateCSV.py -p ./newds_fvectors_mode -d "${resultsfolders[$i]}" -k "$k" -e /run/media/vincenzo/HDATA/CovellaScript/multinode/DT_N15_mode/N15_DT_100mostImportantFeatures.txt -g no -o no -i dt -l 100 -n 10 | tee "logfile_multirun_fixedclf_${k}.log"
	echo "Finished experiment $k"
done
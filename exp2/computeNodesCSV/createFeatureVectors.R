#!/usr/bin/env Rscript

#2020-06-26 17:40
#author: Vito Vincenzo Covella

library("data.table")
library("progress")
library("dplyr")
library("argparser")
library("zoo")
library("bit64")


getMostRecentFault = function(vec){
	s <- sum(vec)
	#if everything is labeled with "healthy" (label 0), the sum is 0, return 0
	if(s == 0){
		return(0)
	} else {		#else return the most recent fault label
		current_label <- 0
		for (i in 1:length(vec))
		{
			if (vec[i] != 0) {
				current_label <- vec[i]
			}
		}

		return(current_label)
	}
}

#Get the sum of the changes within the array-like object
getSumChanges = function(vec){
	sum(diff(vec))
}

p <- arg_parser("Program to create feature vectors for each compute node")
p <- add_argument(p, "--window" , help="window size", default=60)
p <- add_argument(p, "--feature" , help="number of features per each metric", default=6)
p <- add_argument(p, "--step" , help="step size", default=1)
argv <- parse_args(p)

if(argv$feature != 6 & argv$feature != 11)
{
	stop("feature parameter must be 6 or 11.")
}


#gets a filename lists of all csv that do not contain the character '_'
file_list <- list.files(path=getwd(), pattern="\\b([^_]+\\.)csv")

for (k in 1:length(file_list))
{
	print(paste("Reading from disk file: ", file_list[k]))
	orig_df <- fread(file_list[k], header=TRUE)
	#drop all columns with name "cpuXX/<metricname>"
	orig_df <- select(orig_df, -matches('cpu[0-9]'))
	orig_df <- select(orig_df, -contains('applicationLabel'))
	orig_df <- select(orig_df, -contains('faultPred'))
	orig_df <- select(orig_df, -Time)
	#change faultLabel in label
	setnames(orig_df, "faultLabel", "label")

	new_df <- data.table()

	orig_col_length = length(colnames(orig_df))
	columns <- colnames(orig_df)

	print("Computing new dataframe")

	#create progress bar
	pb <- progress_bar$new(format = " computing [:bar] :percent eta: :eta",total = orig_col_length, clear = FALSE, width=100)
	pb$tick(0)
	#loop over columns
	#for each column, calculate the indicators defined in the LRZ report of March 2020
	for (c in 1:orig_col_length)
	{
		#update progress bar
		pb$tick()

		df_col <- select(orig_df, columns[c])
		if (columns[c] != "label")
		{
			ans <- data.table(rollapply(df_col, argv$window, FUN=mean, by=argv$step))
			setnames(ans, columns[c], paste0("mean_", columns[c]))
			new_df <- cbind(new_df, ans)

			ans <- data.table(rollapply(df_col, argv$window, FUN=sd, by=argv$step))
			setnames(ans, columns[c], paste0("std_", columns[c]))
			new_df <- cbind(new_df, ans)

			if (argv$feature == 11){
				ans <- data.table(rollapply(df_col, argv$window, by=argv$step, FUN="quantile", p=0.05))
				setnames(ans, columns[c], paste0("perc5_", columns[c]))
				new_df <- cbind(new_df, ans)
			}

			ans <- data.table(rollapply(df_col, argv$window, by=argv$step, FUN="quantile", p=0.25))
			setnames(ans, columns[c], paste0("perc25_", columns[c]))
			new_df <- cbind(new_df, ans)

			ans <- data.table(rollapply(df_col, argv$window, by=argv$step, FUN="quantile", p=0.75))
			setnames(ans, columns[c], paste0("perc75_", columns[c]))
			new_df <- cbind(new_df, ans)

			if (argv$feature == 11){
				ans <- data.table(rollapply(df_col, argv$window, by=argv$step, FUN="quantile", p=0.95))
				setnames(ans, columns[c], paste0("perc95_", columns[c]))
				new_df <- cbind(new_df, ans)
			}

			ans <- data.table(rollapply(df_col, argv$window, by=argv$step, FUN=getSumChanges))
			setnames(ans, columns[c], paste0("sumdiff_", columns[c]))
			new_df <- cbind(new_df, ans)

			ans <- data.table(rollapply(df_col, argv$window, by=argv$step, FUN="tail", n=1))
			setnames(ans, columns[c], paste0("last_", columns[c]))
			new_df <- cbind(new_df, ans)

			if (argv$feature == 11){
				ans <- data.table(rollapply(df_col, argv$window, by=argv$step, FUN=min))
				setnames(ans, columns[c], paste0("min_", columns[c]))
				new_df <- cbind(new_df, ans)

				ans <- data.table(rollapply(df_col, argv$window, by=argv$step, FUN=max))
				setnames(ans, columns[c], paste0("max_", columns[c]))
				new_df <- cbind(new_df, ans)

				ans <- data.table(rollapply(df_col, argv$window, by=argv$step, FUN=median))
				setnames(ans, columns[c], paste0("median_", columns[c]))
				new_df <- cbind(new_df, ans)

			}

		} else {

			ans <- data.table(rollapply(df_col, argv$window, by=argv$step, FUN=getMostRecentFault))
			new_df <- cbind(new_df, ans)

		}

	}

	#put label as last column
	new_df <- relocate(new_df, "label", .after=last_col())

	print("Saving dataframe")
	fwrite(new_df, paste0(substr(file_list[k], 1, nchar(file_list[k])-4), "_", argv$feature, "f_", argv$window, "s_", argv$step, "step.csv"))
}
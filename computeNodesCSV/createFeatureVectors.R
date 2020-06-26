#2020-06-26 17:40
#author: Vito Vincenzo Covella

library("data.table")
library("progress")
library("dplyr")

#gets a filename lists of all csv that do not contain the character '_'
file_list <- list.files(path=getwd(), pattern="\\b([^_]+\\.)csv")

for (k in 1:length(file_list))
{
	orig_df <- fread(file_list[k], header=TRUE)
	#drop all columns with name "cpuXX/<metricname>"
	orig_df <- select(orig_df, -matches('cpu[0-9]'))
	orig_df <- select(orig_df, -contains('applicationLabel'))
	orig_df <- select(orig_df, -contains('faultPred'))
	orig_df <- select(orig_df, -Time)
	#TODO: complete this code

}
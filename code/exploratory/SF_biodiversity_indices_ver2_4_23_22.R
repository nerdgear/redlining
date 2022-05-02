#######Chapter 1: SF Ebird Biodiversity#######
##Last Updated 5.2.22###########
#Created by Oliver Nguyen (oliverx@stanford.edu or einna.oku@gmail.com)
 

####### Part 1: Subsetting ###############
##Pull SF ebird map (SF_ebird_3.16.22) and subset data into historical (1880-1999) and modern (2000-2022)

setwd("C:/Users/The Reckoning/Documents/Stanford/Data/") #set working directory

SF_ebird_3_16_22 <- read.csv("~/Documents/Stanford/Data/SF_ebird_3_16_22.csv", header=TRUE)

#subset data into historical (1880-1999) and modern (2000-2022)
#we're going to need to split the OBSERVATION.DATE 
#install.packages("splitstackshape")
library(splitstackshape)
Ebird_split_duration <- cSplit(SF_ebird_3_16_22, "OBSERVATION.DATE", "/")

#Check the table. For some reason it had the time on it too, we only want the year 
Ebird_split_duration$OBSERVATION.DATE_3 <- substr(Ebird_split_duration$OBSERVATION.DATE_3, 0, 4)

Ebird_hist <-  subset(Ebird_split_duration, (OBSERVATION.DATE_3 >= 1800 & OBSERVATION.DATE_3 <= 2012)) 
Ebird_mod <- subset(Ebird_split_duration, (OBSERVATION.DATE_3 >= 2012 & OBSERVATION.DATE_3 <= 2022)) 

#export out datset to spatially join to HOLC data in ArcGIS

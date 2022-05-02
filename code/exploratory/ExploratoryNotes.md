# Notes

This is to keep track of the changes and notes made for the project. 

##Updates
### 3.16.22
1. Used raw SF Ebird Data and turned into XY 
2. Projected to UTM NAD 1985 Zone 10N
3. Projected Bay area county (SF county only) to UTM Zone 10N
4. Clipped SF ebird data to Bay Area County (file is called SFclip proj)
5. Above all steps were done in ArcGIS Pro

# 3.25.22
- Pulled SF ebird map (called SF_ebird_3.16.22) onto R studio and separated out data sets (1880-1990) and (2000-2022) to reduce computation time
- Spatially joined bird data points to redlined data HOLC for each data point 

#5.2.22
-Had some problems runnign exploratory analysis with 2 million eBird data points for SF
-Had a meeting with Kelley Langhans and Suzanne Ou, talked about narrowing down questions. 
-Meeting notes were taken on whiteboard. Uploaded pictures in Exploratory notes called 'Meeting_4_26_22'
-Narrowed down questions and will be pursuing first question
	1. How does redlining differ in redlining communities vs. non-redlined communities
	
-will have to filter eBird data
	1. Sampling Efforts
	2. Number of checklist per species (more than 10)
	3. Filter from 2010-2022

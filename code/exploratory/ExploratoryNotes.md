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

#5.1.22
-Had some problems runnign exploratory analysis with 2 million eBird data points for SF
-Had a meeting with Kelley Langhans and Suzanne Ou, talked about narrowing down questions. 
-Meeting notes were taken on whiteboard. Uploaded pictures in Exploratory notes called 'Meeting_4_26_22'
-Narrowed down questions and will be pursuing first question
	1. How does redlining differ in redlining communities vs. non-redlined communities
	
-will have to filter eBird data
	1. Sampling Efforts
	2. Number of checklist per species (more than 10)
	3. Filter from 2010-2022

#5.2.22

##Here are some notes to ask Eric Wood
Questions to ask:
-Time scale on eBird?
-How to grab multiple ebird data?
-What kinds of birds to use?

##Code 
-Cloned github repo from gchure/reproducible research onto VScode 
-Changed origin remote to save to existing repo in Github/nerdgear/redlining:
Code is here:
https://stackoverflow.com/questions/51979534/how-can-i-change-the-origin-remote-in-vscode

```
//1. Go to the root of the directory.

//2. List your existing remotes in order to get the name of the remote you want to change.

$ git remote -v
origin  git@github.com:USERNAME/REPOSITORY.git (fetch)
origin  git@github.com:USERNAME/REPOSITORY.git (push)
Change your remote's URL from SSH to HTTPS with the git remote set-url command.

$ git remote set-url origin https://github.com/USERNAME/REPOSITORY.git

//3. Verify that the remote URL has changed.

$ git remote -v
origin  https://github.com/USERNAME/REPOSITORY.git (fetch)
origin  https://github.com/USERNAME/REPOSITORY.git (push)
```
##More notes
-updated R code to filter for just years. Still over 1 million observations.
-Used this guide to further look for filtering: https://cornelllabofornithology.github.io/ebird-best-practices/ebird.html
-For some reason, ebird modern data filtered only has 2012, even though its been filtered for 2010

#5.3.22
-Updating R code
-Sucessfully filtered and cut down ebird data
	1. using only latest data since 2010
	2. Using more than 10 checklists
	3. controlling for more than 5 km distance travelled 
	4. Cut down variables too 

	Code here: 

	```
	ebd_mod_filtered <- Ebird_mod %>% 
  filter(
    # effort filters
    DURATION.MINUTES <= 5 * 60,
    EFFORT.DISTANCE.KM <= 5,
    # 10 or fewer observers
    NUMBER.OBSERVERS <= 10)

#Keep only necessary columns
ebird <- ebd_mod_filtered %>% 
  select (OBSERVER.ID, SAMPLING.EVENT.IDENTIFIER,
         SCIENTIFIC.NAME, COMMON.NAME,
         OBSERVATION.COUNT, 
         STATE.CODE, LOCALITY, LOCALITY.ID, LATITUDE, LONGITUDE,
         PROTOCOL.TYPE, ALL.SPECIES.REPORTED, Year, Month, Day,
         TIME.OBSERVATIONS.STARTED, 
         DURATION.MINUTES, EFFORT.DISTANCE.KM,
         NUMBER.OBSERVERS)
	```
-uploading SF HOLC and using this link:
https://epsg.io/7131#:~:text=NAD83(2011)%20%2F%20San%20Francisco%20CS13%20%2D%20EPSG%3A7131
-ESPG 7131
-made it up to clipping to sf and removing water from census tracts
-there's a hiccup in that the program can't read the attribute table

#5.5.22

##Meeting with Gretchen 
-Greenspace break it down, make it more accessible 
-not all greenspace is equal, make an analysis of all greenspaces
	-but also greenspaces that are parks
	-habitat suitable greenspaces
-non-native trees or not
-preference for native species
-vegetation structure across different cities 
-LiDAR looking at connectivity
-does looking at vegetation structure differ between areas of the community v. redlined urban spaces?
-Relation to bird community 
-Maybe looking at density and socioeconomic injustice 
-Google around for this 
-wealthy areas in UK (cm tall)
Johnny Hues (wild things)

-Indicate what questions you want to answer
-you can mock up the graphs
-time to analyze the data
-mock up sizes of redlined areas
-biodiversity metric on y axis, 
-second question get a few pictures of connectivity and contrasting of greenspaces with amount of vegetation in them
-describe what the question would be, how sensitive they are to vegetations structure, benefits to people
-Get 5 years of data and work with the first comment 
-throw the draft of the slides 

#5.6.22

##meeting with Kelley
Increase greenspace in low income neighborhoods, define as some way
modelling distributionof greenspaces in different scenarios, these are reasons why scenarios are dnagerous
-gentrification, bird biodiversity, 
we shouldn't add large parks or community gardnes, 
specific types of greenspaces
-compare scenarios
-assigned landuse model, design scenarios bird biodivesity
-talk to people in NatCap!!! 
-What do they take into account (Rafa, Adrien, Andrea, Hector), basically anybody in NatCap
-Models based off landuse (future or landuse scenarios)

1. Look at how to engage communities, and build scenarios 
2. Make a scenario-based modelling, greenspaces (what kind of greenspace), and how to distribute, how it would impact biodiversity







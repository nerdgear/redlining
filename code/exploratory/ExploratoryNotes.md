# Notes

This is to keep track of the changes and notes made for the project. 

##Updates
### 3.16.22
1. Used raw SF Ebird Data and turned into XY 
2. Projected to UTM NAD 1985 Zone 10N
3. Projected Bay area county (SF county only) to UTM Zone 10N
4. Clipped SF ebird data to Bay Area County (file is called SFclip proj)
5. Above all steps were done in ArcGIS Pro

### 3.25.22
- Pulled SF ebird map (called SF_ebird_3.16.22) onto R studio and separated out data sets (1880-1990) and (2000-2022) to reduce computation time
- Spatially joined bird data points to redlined data HOLC for each data point 

### 5.1.22
-Had some problems runnign exploratory analysis with 2 million eBird data points for SF
-Had a meeting with Kelley Langhans and Suzanne Ou, talked about narrowing down questions. 
-Meeting notes were taken on whiteboard. Uploaded pictures in Exploratory notes called 'Meeting_4_26_22'
-Narrowed down questions and will be pursuing first question
	1. How does redlining differ in redlining communities vs. non-redlined communities
	
-will have to filter eBird data
	1. Sampling Efforts
	2. Number of checklist per species (more than 10)
	3. Filter from 2010-2022

### 5.2.22

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

###5.3.22
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

###5.5.22

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


###5.10.22
-finally got the HOLC code to work and spatially join with SF ebird data
-will have to move data to summarize for redlining blocks

##5.16.22
-Working on slides via Canvas
-created a diversity column for ebird in R 
-need to work on standardizing areas for HOLC codes
-ND= non-designated areas for HOLC (non-redlined areas)
-Imported GEE SF NDVI to ArcGIS
-mgight be useful

##6.16.22
-Need to prepare for AOS 
1. Read updated paper and respond to Chloe and Dr. Wood's email	
2. Email Gretchen plan for rest of the summer
3. Make slides in Spanish for AOS

What is novelty? 
-Figure out sampling efforts
-Look at distibution of native vs. non-native guilds

Why use GBIF?


Graph out efforts for each of the areas
from Ellis Soto et al 2022; https://cran.r-project.org/web/packages/KnowBR/index.html
KnowBR shows sampling efforts	


##6.20.22
-worked on looking at the how efforts are placed throughout the city, seems like there are more checklists being done in 
C designated areas
-next steps are 
1. Look at bird communities (see Hensley et al 2019 https://www.frontiersin.org/articles/10.3389/fevo.2019.00071/full#B1)
for bird trait charts from Cornell
	-there's some avibase API data and cornell lab API
2. Look at gentrification maps, see distribution of wealth overtime and maybe make a mixed model effect of the variables
3. What kinds of greenspace are available here?
4. Look into greenspace distribution (read Eric's paper on Greenspace distribution)

#Meeting with Eric

##6.22.22
-Created greenspace from ArcGIS using NDVI map generated from Google Earth
-Used Zonal Statistics as Table and used mean values for each redlined polygon 
Used this paper as a guide: 
https://ehp.niehs.nih.gov/doi/10.1289/EHP7495
-Created NDVI box plot for each redlined area
-add income map

##6.23.22
-Travelling in route to AOS! Working in the airport right now
-currently need to cut down slides, and add in bird community
-maybe look at functional diversity indicies 
-definately include species richness! 

##6.24.22
-Getting some confusing results for species richness, used log mins to break down the efforts and control for that
-started creating a rough list of native vs non native species, will want to look at a couple of bird species

##6.25.22
https://www.nps.gov/prsf/learn/nature/presidio-birds.htm

##7.26.22
-Using Roccupy to look at multi-species occupancy to address observer bias in birds

#7.28.22
Currently trying to use USGS DEP project to download LiDAR data for vegetation in San Francisco
-Downloading Roccupy using this link:
https://github.com/martiningram/roccupy

Going to try to work on downloading LiDAR data from USGS

Other LiDAR sources are linked in resource folder, tried asking EarthDefine for DSM model, and they hav LiDAR data from 2012
But it will cost over $1000 dollars! yeesh

#8.1.22
Had some problems with Jax, error shows in occupancymodelling.R code and the workthrough is documented there and here:
#we get an error here since we're a window user for JAX:
# ERROR: Could not find a version that satisfies the requirement jaxlib>=0.1.62 (from numpyro) (from versions: none)
# ERROR: No matching distribution found for jaxlib>=0.1.62

#So we need to download manually:
#https://pypi.org/project/gpmp/#:~:text=To%20install%20jaxlib%20on%20Windows,Then%20install%20jax%20manually.

#I opened up gitbash and used this code: pip install jax==0.3.13 https://whls.blob.core.windows.net/unstable/cuda111/jaxlib-0.3.7+cuda11.cudnn82-cp38-none-win_amd64.whl

Also I documented the code in the Notebook: "gitbash_jaxerrordocuemtn>8.2.22"

#8.2.22
Still working on introduction
Still running into issues with JAX same error shows up
Some next steps would be to:
1. Reach out for help with Roccupy maybe with Kyle or Sam
2. Work on vegetation complexity
3. Clean up ebird data
	Filter based on Matt Stimas reccomendations + use the data
	Clean out to exclude pelagic birds too


To install CFO, follow this video to set up the envrionemtn
https://www.youtube.com/watch?v=ThU13tikHQw
and to install libraries 

#8.5.22
So nothing helping with this error, I emailed Dr. Ingram to see what will ekp. 
If nothing else ebird has a a code on occupancy modeling which I can use and then 
 I can move onto CFO for vegetation space complelxity

#8.6.22
-Dr. Ingram got back to me (I should be sure to acknowledge his package in the paper)
Moving onto CFO for vegetation space complexity 


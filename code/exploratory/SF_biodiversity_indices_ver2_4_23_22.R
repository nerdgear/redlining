#######Chapter 1: SF Ebird Biodiversity#######
##Last Updated 5.2.22###########
#Created by Oliver Nguyen (oliverx@stanford.edu or einna.oku@gmail.com)
#This version lives as an exisitng exploratory code in github/nerdgear/redlining/exploratory
 

####### Part 1a: Filter year ###############
##Pull SF ebird map (SF_ebird_3.16.22) and subset data into historical (1880-1999) and modern (2000-2022)

setwd("C:/Users/The Reckoning/Documents/Stanford/Data/") #set working directory

#Be sure to update this, the latest eBird downloaded was from Sept 2021
SF_ebird_2021 <- read.delim("~/Documents/Stanford/Data/Ebird_SF_Sept2021/ebd_US-CA-075_relSep-2021.txt", header=TRUE, sep = "\t")


#subset data into historical (1880-2009) and modern (2010-2022)
#we're going to need to split the Year
library(dplyr)
library(tidyr)

SF_ebird_split <- separate(SF_ebird_2021, "OBSERVATION.DATE", c("Year", "Month", "Day"), "-")


Ebird_hist <-  subset(SF_ebird_split, (Year <= 2009)) 
Ebird_mod <- subset(SF_ebird_split, (Year >= 2010)) 

#Gretchen's data suggestion get 5 year data

#We're planning to just use Ebird_mode, but there's still quite a lot of eBird data (over 1 million observations)
#For some reason, it only has 2012 as the oldest eBird observation should look into that


####### Part 1b: Filter Checklists #############
#This part of the code was taken from Cornell lab's best eBird practices 

#load in libraries
library(auk)          #this is a package to created by cornell to use with ebird
library(lubridate)
library(sf)
library(gridExtra)
library(tidyverse)

# function to convert time observation to hours since midnight
time_to_decimal <- function(x) {
  x <- hms(x, quiet = TRUE)
  hour(x) + minute(x) / 60 + second(x) / 3600
}

# clean up variables
Ebird_mod <- Ebird_mod %>% 
  mutate(
    # convert X to NA
    OBSERVATION.COUNT = if_else(OBSERVATION.COUNT == "X", 
                                NA_character_, OBSERVATION.COUNT),
    OBSERVATION.COUNT = as.integer(OBSERVATION.COUNT),
    # effort_distance_km to 0 for non-travelling counts
    EFFORT.DISTANCE.KM = if_else(PROTOCOL.TYPE != "Traveling", 
                                 0, EFFORT.DISTANCE.KM),
    # convert time to decimal hours since midnight
   TIME.OBSERVATIONS.STARTED= time_to_decimal(TIME.OBSERVATIONS.STARTED),
  )


# additional filtering
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


######## Part 2: Exploratory Analysis #############
#Now we're going to load in HOLC data for SF and see how it compares


# load and project HOLC GIS Data
#We're going to use the gis-data.pckg that ebird provides as a baselayer


map_proj <- st_crs(7131) #we're using projection specifcally for SF

#Now we're going to load in base layesr using Tigris 
#install.packages('tigris')
library(tigris)
library(dplyr)
library(sp)
library(ggplot2)

# SF_bg <- block_groups("CA", "San Francisco")


CA_county <- counties("CA")
#subset for just SF county

SF_county  <- CA_county[CA_county$NAME == "San Francisco",]
 ggplot() +
   geom_sf(data = SF_county) +
   theme_void()
 
#Create a function to remove water from tracts
#install.packages("rmapshaper")
library(rmapshaper)

 get_tracts <- function(state_abbr,
                        county_name,
                        year=NULL, 
                        refresh=FALSE, 
                        remove_water=TRUE){
   print(paste0("getting shapefile for tracts in ",state_abbr))
   state_tracts <- tracts(cb = TRUE, 
                          year = year, 
                          class="sf", 
                          state=state_abbr,
                          county=county_name,
                          refresh=refresh) %>% 
     st_transform(., crs=4326) %>% st_make_valid()
   Sys.sleep(1) #included for politeness to census servers
   if(!remove_water){
     return(state_tracts)
   }
   print(paste0("fetching water boundaries for ",state_abbr))
   
   county_codes <- unique(state_tracts$COUNTYFP)
   
   print(paste0("joining water areas into one shapefile for ",state_abbr))
   
   # https://community.rstudio.com/t/better-ways-to-remove-areas-of-water-from-us-map/60771
   # https://gis.stackexchange.com/questions/308119/function-in-r-to-get-difference-between-two-sets-of-polygons-comparable-in-speed
   water_area <- 
     map_dfr(
       county_codes,
       ~ area_water(state_abbr, 
                    county = ., 
                    year = year,
                    refresh = refresh,
                    class = "sf")  %>% 
         select("geometry") %>% 
         st_transform(., crs=4326) %>% 
         st_make_valid()
     )
   
   print("subtracting water areas from census tract areas")
   
   state_tracts_sans_water <- ms_erase(ms_simplify(state_tracts),
                                       ms_simplify(water_area), 
                                       remove_slivers=TRUE,
                                       sys=FALSE)
   Sys.sleep(1)
   return(state_tracts_sans_water)
 }
 
 #test the function
 sf_tracts <- get_tracts("CA", "San Francisco", year=2020,remove_water = FALSE)
 sf_tracts %>%
   ggplot(aes(fill = STATEFP)) +
   geom_sf() 
 
 
#Note to self, might be worth creating a function that gets rid of islands too
 
#bring in other census data
#  SF_bg_prj <- SF_bg %>% 
#    st_transform(crs = map_proj) %>% 
#    st_geometry()


#HOLC redlining data
library(rgdal)

#for some reason readOGR works better than st_read for HOLC data
SF_HOLC<- readOGR("~/Documents/Stanford/Data/redline_shapefiles/CASanFrancisco1937", "cartodb-query") %>%
st_as_sf(SF_HOLC) %>%
st_transform(SF_HOLC, crs=map_proj)
names(SF_HOLC) # check the columns
class(SF_HOLC) #check the class (should be sf in order for the data to work)

ggplot() +
  geom_sf(data = SF_HOLC) +
  theme_void()


# prepare ebird data for mapping
ebird_sf <- ebird %>% 
  # convert to spatial points
  st_as_sf(coords = c("LONGITUDE", "LATITUDE"), crs = 4326) %>% 
  st_transform(crs = map_proj) 

# map
par(mar = c(0.25, 0.25, 0.25, 0.25))
# set up plot area
#plot(st_geometry(ebird_sf))

# contextual gis data
plot(SF_HOLC, col = "#dddddd", border = "#888888", lwd = 0.5, add = TRUE)
#plot(SF_bg_prj, col = "#ffffff", lwd = 0.75, add = TRUE)


#seems like there's some ebird observations that are outside of our intended area, so we should clip to SF county
#let's use Sf_tracts to remove any ebird data that isn't on land

#check if crs are same
st_crs(ebird_sf)==st_crs(SF_HOLC)

#if true then we can continue, ignore the code below, this was just to grab only points that fell within the redlined areas
#ebird_clip<-st_crop(ebird_sf, SF_HOLC)

#spatial join
#ebird_HOLC <- st_join(ebird_clip, left = FALSE, SF_HOLC["holc_grade"]) # join points

#standardize land + calculate total geographic area of HOLC 
SF_HOLC$area <- st_area(SF_HOLC)
library(units)
SF_HOLC$area_km <- set_units(SF_HOLC$area, km^2)

#get rid of the units 
attributes(SF_HOLC$area_km) = NULL

#this grabs ebird for all redlined and non-redlined areas
ebird_HOLC_all <- st_join(ebird_sf, left = TRUE, SF_HOLC["holc_grade"]) #join points 

#create a new table with diversity indices
#install.packages("vegan")
library(vegan)

#we need to transform the table from long to wide format
#install.packages("reshape")
library(reshape)
ebird_table <- as.data.frame(ebird_HOLC_all) #convert sf object into dataframe so we can reshape it
#Create NA class for HOLC (ND-- Non-designated)
ebird_table["holc_grade"][is.na(ebird_table["holc_grade"])] <- "ND"

#summarize observation counts for each HOLC grade 
ebird_HOLC_agg <- aggregate(OBSERVATION.COUNT~COMMON.NAME+holc_grade, data=ebird_table, FUN=sum) 
ebird_reshape <- reshape(ebird_HOLC_agg, idvar = "holc_grade", timevar = "COMMON.NAME", direction = "wide")
#ebird_reshape <- reshape(ebird_HOLC_agg, idvar = "COMMON.NAME", timevar = "holc_grade", direction = "wide")


#check that we actually have all the unique species in there
length(unique(ebird_table[["COMMON.NAME"]])) #488 species matches up with 488 columns made on the table

#replace NA with 0 (species that were not seen at all that year/time)
ebird_reshape[is.na(ebird_reshape)] <- 0

library(vegan)
ebird_reshape$shannons <- diversity(ebird_reshape[-1], index="shannon")


#plot out shannons
library(ggplot2)

#HOLC colors
holc_colors = c("#6495ED", "#9FE2BF", "#FFBF00", "#DE3163")
ggplot(data = ebird_reshape[-c(5),], aes(x = holc_grade, y = shannons, group = 1)) +
   labs(x="HOLC ID",y= "Shannon's Diversity") +
   geom_point(size = 4, color= holc_grade, show.legend = FALSE) +
   scale_color_manual(values = c("A" = "#6495ED", "B" = "#9FE2BF"))

pe+ scale_fill_manual(values=c("#6495ED", "#9FE2BF", "#FFBF00", "#DE3163"))

   




#This might be next step is to figure out what birds    
ebird_reshape_names <- make.names(names(ebird_reshape))
ggplot(data = ebird_reshape, aes(x = holc_grade, y = House.Sparrow  )) + labs(x="HOLC ID",y= "House Sparrow Observation Counts") +
   geom_bar()

?subset

#Look at observer effort between different zones
# install.packages("KnowBR")
# library(KnowBR)

#Look at Sampking Event Identifer and look at counts in each HOLC grade
samplingffort <- ebird_HOLC_all %>%                              # Applying group_by & summarise
   group_by(holc_grade) %>%
   summarise(count = n_distinct(SAMPLING.EVENT.IDENTIFIER, na.rm = TRUE)) 

#remove NA for now 
ggplot(data= samplingffort[-c(5),], aes(x = holc_grade, y = count)) + labs(x="HOLC ID",y= "Number of Checklists") +
   geom_bar(stat='identity')


#Seems like C has a lot, we should control for area size

#now control for area size
HOLC_area <- aggregate(SF_HOLC$area_km, by=list(holc_grade=SF_HOLC$holc_grade), FUN=sum)
#graph this out 
ggplot(data= HOLC_area[-c(5),], aes(x = holc_grade, y = x, fill= holc_grade)) + labs(x="HOLC ID",y= "Area Size (km)") +
   geom_col() + scale_fill_manual(values=c("#9FE2BF", "#6495ED", "#FFBF00", "#DE3163"))


#combine checklist by area
HOLC_areaeffort <- merge(x=HOLC_area,y= samplingffort, by="holc_grade")

HOLC_areaeffort$area_effort <- HOLC_areaeffort$count/HOLC_areaeffort$x

#graph
ggplot(data= HOLC_areaeffort, aes(x = holc_grade, y = area_effort)) + labs(x="HOLC ID",y= "Checklists Divided by Area (km2)") +
   geom_bar(stat='identity')

 
#Now take a look at minutes observation                           
samplingmin <- aggregate(ebird_HOLC_all$DURATION.MINUTES, by=list(holc_grade=ebird_HOLC_all$holc_grade), FUN=sum)
ggplot(data= samplingmin, aes(x = holc_grade, y = x)) + labs(x="HOLC ID",y= "Duration Minutes") +
   geom_bar(stat='identity')


######### Part 2a. Controlling for efforts and area ############
#log minutes since it's a big number to work with
samplingmin$logmin <- log(samplingmin$x)

#merge minutes and shannons diversity 
HOLC_merge_min <- merge(samplingmin, ebird_reshape, by = "holc_grade")

require(dplyr)
HOLC_merge_min <-HOLC_merge_min%>%
   mutate(diversemin=shannons/logmin)

#plot it out
ggplot(data = HOLC_merge_min, aes(x = holc_grade, y = diversemin, group = 1)) + labs(x="Years", y="Shannon's Divesity/logmin") +
   geom_point() + geom_line()

#let's get some raw richness, oh yeahhh
library(plyr)


SF_richness_raw <- ddply(ebird_reshape,~holc_grade,function(x) {
   data.frame(RICHNESS=sum(x[-1]>0))
})

ggplot(data= SF_richness_raw[-c(5),], aes(x = holc_grade, y = RICHNESS, group = 1)) + labs(x="HOLC ID",y= "Species Richness") +
   geom_point(size=4)

#divide by observer hours by species richness

#merge minutes and shannons diversity 
HOLC_merge_min_rich <- merge(samplingmin, SF_richness_raw, by = "holc_grade")

require(dplyr)
HOLC_merge_min_rich <-HOLC_merge_min_rich%>%
   mutate(richmin=RICHNESS/logmin)

#plot it out
ggplot(data = HOLC_merge_min_rich, aes(x = holc_grade, y = richmin, group = 1)) + labs(x="Years", y="Species Richness/logmin") +
   geom_point() + geom_line()

############Part 3: NDVI Analysis ###########
#Import NDVI chart
SF_NDVI <- SF_HOLC_NDVI_stats
# Change box plot line colors by groups
p<-ggplot(SF_NDVI, aes(x=holc_grade, y=sf_zonalstats_ndvi_MEAN, fill=holc_grade)) +
   scale_y_continuous(trans = "reverse") +
   geom_boxplot() +
   xlab("HOLC Grade") +
   ylab("NDVI Mean") 
p+ scale_fill_manual(values=c("#9FE2BF", "#6495ED", "#FFBF00", "#DE3163"))

#compare means 
#install.packages("ggpubr")
#install.packages("car")
library(ggpubr)
library(car)
compare_means(sf_zonalstats_ndvi_MEAN ~ holc_grade,  data = SF_NDVI, method = "anova")

###########Part 4: Bird Community Species #################
#remove OBservatoin count from colnames from ebird_reshape
colnames(ebird_reshape)<-gsub("OBSERVATION.COUNT.","",colnames(ebird_reshape))


#rename the columns and get rid of spaces, so that now that's . in spaces and places with commas
#example: Allen's Hummingbird becomes Allen.s.Hummingbird
names(ebird_reshape)<-make.names(names(ebird_reshape),unique = TRUE)
#this above code replaces spaces, ', and - with periods (.)


HOLC_merge_min_pigeon <-HOLC_merge_min%>%
   mutate(pigmin=Rock.Pigeon/logmin)

ggplot(data= ebird_reshape[-c(5),], aes(x = holc_grade, y = Rock.Pigeon, fill= holc_grade)) + labs(x="HOLC ID",y= "Rock Pigeon") +
   geom_col() + scale_fill_manual(values=c("#9FE2BF","#6495ED", "#FFBF00", "#DE3163"))
   

#pigeon contrlled for minutes
ggplot(data= HOLC_merge_min_pigeon[-c(5),], aes(x = holc_grade, y = pigmin, fill= holc_grade)) + labs(x="HOLC ID",y= "Rock Pigeon Counts/logmin") +
   geom_col() + scale_fill_manual(values=c("#9FE2BF","#6495ED", "#FFBF00", "#DE3163"))


#wilson's warbler
ggplot(data= ebird_reshape[-c(5),], aes(x = holc_grade, y = Wilson.s.Warbler, fill= holc_grade)) + labs(x="HOLC ID",y= "Wilson's warbler") +
   geom_col()

HOLC_merge_min_warbler <-HOLC_merge_min%>%
   mutate(warbmin= Wilson.s.Warbler/logmin)

ggplot(data= HOLC_merge_min_warbler[-c(5),], aes(x = holc_grade, y = warbmin, fill= holc_grade)) + labs(x="HOLC ID",y= "Wilson's warbler/logmin") +
   geom_col() +  scale_fill_manual(values=c("#6495ED", "#9FE2BF", "#FFBF00", "#DE3163"))



#we're going to do like a group of them just to see whats up
#subset for the following:

selectbirds <- c(
'Rock.Pigeon',
'Red.Tailed.Hawk',
'European.Starling',
'Oregon.Junco',
'White.crowned.Sparrow',
'Black.Phoebe',
'House.Sparrow',
'Red.masked.Parakeet',
'Eurasian.Collared.Dove',
'Mourning.Dove')


#replace all spaces and special charcters in common name with .
ebird_HOLC_replace <- ebird_HOLC_all #make a copy in case we mess up 

#replace all white space
library (magrittr)
library(stringi)

ebird_HOLC_replace$COMMON.NAME <- stri_replace_all_regex(ebird_HOLC_replace$COMMON.NAME,
                                  pattern=c(" ","-","'"),
                                  replacement=c('.', '.', '.'),
                                  vectorize=FALSE)

ebird_HOLC_select <- ebird_HOLC_replace[ebird_HOLC_replace$COMMON.NAME %in% selectbirds,]

ebird_HOLC_native <- merge(Native_nonnativespecies_draft1, ebird_HOLC_select, by.x="Common_Name", by.y= "COMMON.NAME")

#For every HOLC grade, count how many Natives/non-natives there are 

library(plyr)
counts <- ddply(ebird_HOLC_native, .(ebird_HOLC_native$holc_grade, ebird_HOLC_native$Native), nrow)
names(counts) <- c("holc_grade", "Native", "Count")

#graph it out
ggplot(data= counts[-c(9,10),], aes(x = holc_grade , y = Count, fill= Native)) + labs(x="HOLC ID",y= "Count") +
   geom_col(position = "dodge")

#Nice let's graph out different bird guilds
#note to self, need to figure out a better way on how to add parenthesis and make a list for these birds

hab_birds <- c(
'Rock.Pigeon',
'Red.Tailed.Hawk',
'European.Starling',
'Oregon.Junco',
'White.crowned.Sparrow',
'Black.Phoebe',
'House.Sparrow',
'House.Finch',
'Red.masked.Parakeet',
'Eurasian.Collared.Dove',
'Mourning.Dove',
'American.Goldfinch',
'Common.Raven',
'American.Crow',
'Anna.s.Hummingbird',
'Allen.s.Hummingbird',
'Brown.Creeper',
'Wilson.s.Warbler',
'Golden.crowned.sparrow',
'California.Scrub.Jay',
'Stellar.s.Jay',
'Great.Horned.Owl',
'Orange.crowned.Warbler',
'Acorn.Woodpecker',
'Chestnut.backed.Chickadee',
'Bushtit',
'Pine.siskin',
'Hermit Thrush',
'California.Towhee')


#now create selection for birds
   
ebird_HOLC_select <- ebird_HOLC_replace[ebird_HOLC_replace$COMMON.NAME %in% hab_birds,]

ebird_HOLC_habitat <- merge(Habitat_guild_draft1, ebird_HOLC_select, by.x="COMMON.NAME", by.y= "COMMON.NAME")

#count how many there are
count_hab <- ddply(ebird_HOLC_habitat, .(ebird_HOLC_habitat$holc_grade, ebird_HOLC_habitat$Habitat), nrow)
names(count_hab) <- c("holc_grade", "Habitat", "Count")

#graph it out
ggplot(data= count_hab[-c(17,18,19,20),], aes(x = holc_grade , y = Count, fill= Habitat)) + labs(x="HOLC ID",y= "Count") +
   geom_col(position = "dodge")

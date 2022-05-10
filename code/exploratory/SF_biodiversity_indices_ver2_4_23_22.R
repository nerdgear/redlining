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
SF_HOLC<- st_read("~/Documents/Stanford/Data/redline_shapefiles/CASanFrancisco1937", "cartodb-query") %>% 
  st_transform(crs = map_proj) %>% 
  st_geometry()# load and project gis data


#SF_HOLC<- readOGR("~/Documents/Stanford/Data/redline_shapefiles/CASanFrancisco1937", "cartodb-query") 

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

#if true then we can continue
ebird_clip<-st_crop(ebird_sf, SF_HOLC)

#spatial join
ebird_HOLC <- st_join(ebird_clip, left = FALSE, SF_HOLC["holc_grade"]) # join points

#plot(bcr, col = "#cccccc", border = NA, add = TRUE)
#plot(ne_state_lines, col = "#ffffff", lwd = 0.75, add = TRUE)
#plot(ne_country_lines, col = "#ffffff", lwd = 1.5, add = TRUE)


# ebird observations
# not observed
plot(st_geometry(ebird_sf),
     pch = 19, cex = 0.1, col = alpha("#555555", 0.25),
     add = TRUE)
# observed
plot(filter(ebird_sf, species_observed) %>% st_geometry(),
     pch = 19, cex = 0.3, col = alpha("#4daf4a", 1),
     add = TRUE)
# legend
legend("bottomright", bty = "n",
       col = c("#555555", "#4daf4a"),
       legend = c("eBird checklists", "Wood Thrush sightings"),
       pch = 19)
box()
par(new = TRUE, mar = c(0, 0, 3, 0))
title("Wood Thrush eBird Observations\nJune 2010-2019, BCR 27") v  



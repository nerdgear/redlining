# Chapter 1: Redlining


## Introduction 
This is documentation on code used for redlining with ebird data. 

## Methods

### Data
We use data from eBird and Avonet to generate the following datasets. 

The data is split into different groups, totalling 7 differnet datasets stored as comma-seperated files: 

==Data Type== 
A. Habitat Guilds 
    - Urban 
    - Shrub and Grassland
    - Forest
    - Water and Wetlands

B. Native and Non-Native Birds 
    - Native Species
    - Non-Native Species 
C. All Species 


The naming convention for each file will show as follows:  

['City_DataType.csv']

--- 

### Code 

The code is broken up into 6 different parts: 

- [Chapter 1: Redlining](#chapter-1-redlining)
  - [Introduction](#introduction)
  - [Methods](#methods)
    - [Data](#data)
    - [Code](#code)
      - [**Part 1. Cleaning Ebird Data**](#part-1-cleaning-ebird-data)
      - [Part 2. Joining HOLC polygon layers](#part-2-joining-holc-polygon-layers)
      - [Part 3. Running iNEXT on datsets](#part-3-running-inext-on-datsets)
      - [Part 4. Spatially Joining Data Sets](#part-4-spatially-joining-data-sets)
      - [Part 5. Random Forest Modelling](#part-5-random-forest-modelling)



#### **Part 1. Cleaning Ebird Data**
   

This code borrows from eBird best practices and Callaghan et al. 2021 to work with citizen science data, we'll be using auk package to help us clean the data and using R. We downloaded respective datasets from San Francisco and Los Angeles. 


The main filters we will be using are: 

[x] Years 2010 - 2025 (Jan - Dec)  
[x] => 0.5 km distance  
[x] area <50 ha
[x] => 10 observers  
[x] combine shared checklists (Auk_Unique)  
[x] => 15- 120 minutes hours total efforts checklist  
[x] Stationary + Travelling + Area checklists  
[x] Completed only checklists  

We'll be aiming to be using **Incidence Frequency** instead of Abundance (Callaghan et al. 2022) as it would be more stable. 


#### Part 2. Joining HOLC polygon layers

#### Part 3. Running iNEXT on datsets
#### Part 4. Spatially Joining Data Sets
#### Part 5. Random Forest Modelling 



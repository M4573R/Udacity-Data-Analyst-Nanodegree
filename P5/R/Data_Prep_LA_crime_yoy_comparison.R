##################################
### LA Crime by ###
##################################

### Load Packages -------------------------------------------
library(tidyr)
library(plyr)
library(dplyr)
library(lubridate)
library(ggplot2)
library(ggthemes)
library(scales)
library(reshape2)
setwd("~/Documents/Udacity-Data-Analyst-Nanodegree/P5/")
# source("../[201508]Foresee_Score/R/useful_functions.R")

### The BRIGHT and CLEAR self-defined theme function -----------------------------------------
theme_bright_clear <- function(){
    theme_hc() %+replace%
        theme(plot.title = element_text(size = rel(1.8)), 
              axis.text = element_text(size = rel(1.1), colour = "black"),
              axis.title.y = element_text(size = rel(1.3), angle = 90),
              axis.title.x = element_text(size = rel(1.1)),
              legend.text = element_text(size = rel(1.3)),
              legend.title = element_text(size = rel(1.3)))  
}

### Read Data -------------------------------------------

crime_raw_y13 <- tbl_df(read.csv("./Data/LAPD_Crime_and_Collision_Raw_Data_for_2013.csv", as.is = TRUE))
crime_raw_y14 <- tbl_df(read.csv("./Data/LAPD_Crime_and_Collision_Raw_Data_-_2014.csv", as.is = TRUE))

# table(crime_raw_y13$Crm.ctgry)
crime_raw_y13 <- crime_raw_y13 %>%
    mutate(Crm.ctgry = "Others")

### Violet Crime: Aggravated assault, Robbery, Rape, Homicide
### Property Crime: Theft, Burglary, Motor Vehicle Theft
crime_raw_y13$Crm.ctgry[grep("ROBBERY", crime_raw_y13$Crm.Cd.Desc)] <- "Violet Crime" #"Robbery"
crime_raw_y13$Crm.ctgry[grep("AGGRAVATED ASSAULT", crime_raw_y13$Crm.Cd.Desc)] <-  "Violet Crime" #"Aggravated assault"
crime_raw_y13$Crm.ctgry[grep("RAPE", crime_raw_y13$Crm.Cd.Desc)] <-  "Violet Crime" #"Rape"
crime_raw_y13$Crm.ctgry[grep("HOMICIDE", crime_raw_y13$Crm.Cd.Desc)] <-  "Violet Crime" #"Homicide"


crime_raw_y13$Crm.ctgry[grep("THEFT", crime_raw_y13$Crm.Cd.Desc)] <- "Property Crime" #"Theft"
crime_raw_y13$Crm.ctgry[crime_raw_y13$Crm.Cd.Desc == "THEFT OF IDENTITY"] <- "Identity Theft"
crime_raw_y13$Crm.ctgry[grep("BURGLARY", crime_raw_y13$Crm.Cd.Desc)] <- "Property Crime" #"Burglary"
crime_raw_y13$Crm.ctgry[crime_raw_y13$Crm.Cd.Desc %in% c("VEHICLE - ATTEMPT STOLEN", "VEHICLE - STOLEN")] <- "Property Crime" #"Motor Vehicle Theft"

crime_y13_dist_tot <- crime_raw_y13 %>%
    filter(Crm.ctgry %in% c("Violet Crime", "Property Crime")) %>%
    group_by(AREA.NAME, Crm.ctgry)%>%
    summarise(Crm.num.13 = n())



crime_raw_y14 <- crime_raw_y14 %>%
    mutate(Crm.ctgry = "Others")

### Violet Crime: Aggravated assault, Robbery, Rape, Homicide
### Property Crime: Theft, Burglary, Motor Vehicle Theft
crime_raw_y14$Crm.ctgry[grep("ROBBERY", crime_raw_y14$Crm.Cd.Desc)] <- "Violet Crime" #"Robbery"
crime_raw_y14$Crm.ctgry[grep("AGGRAVATED ASSAULT", crime_raw_y14$Crm.Cd.Desc)] <-  "Violet Crime" #"Aggravated assault"
crime_raw_y14$Crm.ctgry[grep("RAPE", crime_raw_y14$Crm.Cd.Desc)] <-  "Violet Crime" #"Rape"
crime_raw_y14$Crm.ctgry[grep("HOMICIDE", crime_raw_y14$Crm.Cd.Desc)] <-  "Violet Crime" #"Homicide"


crime_raw_y14$Crm.ctgry[grep("THEFT", crime_raw_y14$Crm.Cd.Desc)] <- "Property Crime" #"Theft"
crime_raw_y14$Crm.ctgry[crime_raw_y14$Crm.Cd.Desc == "THEFT OF IDENTITY"] <- "Identity Theft"
crime_raw_y14$Crm.ctgry[grep("BURGLARY", crime_raw_y14$Crm.Cd.Desc)] <- "Property Crime" #"Burglary"
crime_raw_y14$Crm.ctgry[crime_raw_y14$Crm.Cd.Desc %in% c("VEHICLE - ATTEMPT STOLEN", "VEHICLE - STOLEN")] <- "Property Crime" #"Motor Vehicle Theft"

crime_y14_dist_tot <- crime_raw_y14 %>%
    filter(Crm.ctgry %in% c("Violet Crime", "Property Crime")) %>%
    group_by(AREA.NAME, Crm.ctgry)%>%
    summarise(Crm.num.14 = n())


crime_y1314_dist_tot <- crime_y14_dist_tot %>%
    left_join(crime_y13_dist_tot, by = c("Crm.ctgry", "AREA.NAME"))

LAPD_crime_1314_sum_to_json <- crime_y1314_dist_tot %>% 
    filter(Crm.ctgry == "Property Crime") %>%
    rename(PC13 = Crm.num.13,
           PC14 = Crm.num.14) %>%
    select(AREA.NAME, PC13, PC14) %>%
    left_join(
        crime_y1314_dist_tot %>% 
            filter(Crm.ctgry == "Violet Crime") %>%
            rename(VC13 = Crm.num.13,
                   VC14 = Crm.num.14) %>%
            select(AREA.NAME, VC13, VC14)
    )


write.csv(LAPD_crime_1314_sum_to_json, "./Data/LAPD_crime_1314_sum_to_json.csv", row.names = FALSE)
######################################
# Data Cleaning and Manipulation
# Author: Rachel Weber
# Created: September 2019
######################################

library(anytime)
library(tidyverse)
library(magrittr)
library(pander)

# read in data
shoot <- read.csv(file = "C:/Users/rache/Documents/Capstone/Data/Historic Data 2006_2018.csv", sep = ",", na.strings=c("","NA"))

shoot$PRECINCT <- as.factor(shoot$PRECINCT)
shoot$OCCUR_DATE <- anytime(shoot$OCCUR_DATE)

# There are dates where no shooting happened
# to know how many and when they occured, I'll fill in any missing dates as empty rows

all_dates <- shoot %>% 
  mutate(OCCUR_DATE = as.Date(OCCUR_DATE)) %>% 
  complete(OCCUR_DATE = seq.Date(min(OCCUR_DATE), max(OCCUR_DATE), by="day"))

# we gained 345 days. There were 345 days between 2006 and 2018 where no shootings happened in NYC
# There are 4748 days between these years
# 7% of days between 2006 and 2018 had no shootings


# let's get a column of just year for shootings
# we know rates have been going down and may have to accomodate that in our analysis
all_dates %<>% 
  mutate(year = lubridate::year(OCCUR_DATE))

# get weekdays
all_dates$day <- weekdays(all_dates$OCCUR_DATE)

getSeason <- function(DATES) {
  WS <- as.Date("2012-12-15", format = "%Y-%m-%d") # Winter Solstice
  SE <- as.Date("2012-3-15",  format = "%Y-%m-%d") # Spring Equinox
  SS <- as.Date("2012-6-15",  format = "%Y-%m-%d") # Summer Solstice
  FE <- as.Date("2012-9-15",  format = "%Y-%m-%d") # Fall Equinox
  
  # Convert dates from any year to 2012 dates
  d <- as.Date(strftime(DATES, format="2012-%m-%d"))
  
  ifelse (d >= WS | d < SE, "Winter",
          ifelse (d >= SE & d < SS, "Spring",
                  ifelse (d >= SS & d < FE, "Summer", "Fall")))
}

all_dates$season <- as.factor(getSeason(all_dates$OCCUR_DATE))

write.csv(all_dates, "C:/Users/rache/Documents/Capstone/Data/NYC_extracolumns.csv")

# let's check that the shooting free weekend is actually here
fall_2018 <- all_dates[all_dates$year == 2018 & all_dates$season == "Fall",]
  # yay Oct 12, 13 & 14 are all NAs
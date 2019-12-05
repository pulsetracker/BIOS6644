########################################
# NYC shooting data project
# Author: Rachel Weber
# Data provided by NYC Open Data
########################################

# the objective of this study is to determine the odds of a shooting-free weekend in NYC
# this was initiated following the first shooting-free weekend in September 2018 NYC has had in a decade
# data were requested and granted from the NYC Open Data resource
# data was received on 11/1/18

library(anytime)
library(CIDAtools)
library(tidyverse)
library(magrittr)
library(pander)
library(ggplot2)

# read in data
shoot <- read.csv(file = "C:/Users/weberra/Documents/NYC Shootings Project/Data/NYPD_Shooting_Incident_Data__Historic_.csv", sep = ",")

shoot$PRECINCT <- as.factor(shoot$PRECINCT)
shoot$OCCUR_DATE <- anytime(shoot$OCCUR_DATE)

# There are dates where no shooting happened
# to know how many and when they occured, I'll fill in any missing dates as empty rows

all_dates <- shoot %>% 
  mutate(OCCUR_DATE = as.Date(OCCUR_DATE)) %>% 
  complete(OCCUR_DATE = seq.Date(min(OCCUR_DATE), max(OCCUR_DATE), by="day"))

# we gained 158 days. There were 158 days in 5 years where no shootings happened in NYC
# assuming 1 leap year in the 5 years of data, there were 1826 days in the dataset
# 8% of days between 2013 and 2017 had no shootings


# let's get a column of just year for shootings
# we know rates have been going down and may have to accomodate that in our analysis
all_dates %<>% 
  mutate(year = lubridate::year(OCCUR_DATE))

# a table 1 to summarize shooting demographic info
table1 <- Table1(c("BORO", "PERP_AGE_GROUP", "PERP_SEX", "PERP_RACE", "VIC_AGE_GROUP", "VIC_SEX", "VIC_RACE"), NULL,
                 data = shoot, incl_missing = F)
set.alignment("center", row.names="left")
pander(table1, caption = "")

# a graph showing number of shootings over the 5 years
ggplot(all_dates, aes(x = factor(year))) + geom_bar() + 
  ggtitle("Number of Shootings by Year") + theme_minimal() +
  xlab("Year") + ylab("N shootings")

# shootings are trending down. Do we need to account for that?
# Since this was the first shooting-free weekend in a decade, why not include all data,
# even from years where rates were higher than the present year?


# get weekdays
all_dates$day <- weekdays(all_dates$OCCUR_DATE)


# get P(no shooting Friday)
pf <- nrow(all_dates[all_dates$day == "Friday" & is.na(all_dates$OCCUR_TIME),])/nrow(all_dates[all_dates$day == "Friday",])
  # .016

# get P(no shooting Saturday)
psat <- nrow(all_dates[all_dates$day == "Saturday" & is.na(all_dates$OCCUR_TIME),])/nrow(all_dates[all_dates$day == "Saturday",])
  # .0032

# get P(no shooting Sunday)
psun <- nrow(all_dates[all_dates$day == "Sunday" & is.na(all_dates$OCCUR_TIME),])/nrow(all_dates[all_dates$day == "Sunday",])
  # .009

# assuming independence, get P(Fri and Sat and Sun)
p_ind_weekend <- pf*psat*psun
  # 4.79e-7

# since we can't really assume independence, we'll use the definition of condition probability to find
# probabilities of each successive day being shooting-free

# P(Sat | Friday) = P(Sat and Fri)*P(Fri)
p_f_sat <- psat*pf*pf
  #8.44e-7

# P(Sun | Sat and Fri) = P(Fri and Sat and Sun)P(Sat and Fri)
p_f_sat_sun <- p_ind_weekend*p_f_sat
  # 4.05e-13

################################### Yearly odds #######################################

# okay so shooting rates have been going down. What I just use 2016 and 2017 data to predict my odds?

y2 <- all_dates[all_dates$year >= 2016,]

# get P(no shooting Friday)
pf <- nrow(y2[y2$day == "Friday" & is.na(y2$OCCUR_TIME),])/nrow(y2[y2$day == "Friday",])
  # .029

# get P(no shooting Saturday)
psat <- nrow(y2[y2$day == "Saturday" & is.na(y2$OCCUR_TIME),])/nrow(y2[y2$day == "Saturday",])
  # .005

# get P(no shooting Sunday)
psun <- nrow(y2[y2$day == "Sunday" & is.na(y2$OCCUR_TIME),])/nrow(y2[y2$day == "Sunday",])
  # .019

# assuming independence, get P(Fri and Sat and Sun)
p_ind_weekend <- pf*psat*psun
  # 2.89e-6

# since we can't really assume independence, we'll use the definition of condition probability to find
# probabilities of each successive day being shooting-free

# P(Sat | Friday) = P(Sat and Fri)*P(Fri)
p_f_sat <- psat*pf*pf
  # 4.38e-6

# P(Sun | Sat and Fri) = P(Fri and Sat and Sun)P(Sat and Fri)
p_f_sat_sun <- p_ind_weekend*p_f_sat
  # 1.27e-11
  # its 2 decimal places of zeros more likely than when calculated using all data (e-11 vs e-13)

############################# but shootings change by the season, let's look at that ############
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

fall <- all_dates[all_dates$season == "Fall",]

# get P(no shooting Friday)
pf <- nrow(fall[fall$day == "Friday" & is.na(fall$OCCUR_TIME),])/nrow(fall[fall$day == "Friday",])
  # .007

# get P(no shooting Saturday)
psat <- nrow(fall[fall$day == "Saturday" & is.na(fall$OCCUR_TIME),])/nrow(fall[fall$day == "Saturday",])
  # 0---There has never in the last 5 years been a saturday without a shooting
  # I can't stratify by season b/c I would be multiplying by 0

# get P(no shooting Sunday)
psun <- nrow(fall[fall$day == "Sunday" & is.na(fall$OCCUR_TIME),])/nrow(fall[fall$day == "Sunday",])
# .007


# What if we look at summer? Usually shootings peak in summer
summer <- all_dates[all_dates$season == "Summer",]

# get P(no shooting Friday)
pf <- nrow(summer[summer$day == "Friday" & is.na(summer$OCCUR_TIME),])/nrow(summer[summer$day == "Friday",])
# .003

# get P(no shooting Saturday)
psat <- nrow(summer[summer$day == "Saturday" & is.na(summer$OCCUR_TIME),])/nrow(summer[summer$day == "Saturday",])
  # .002

# get P(no shooting Sunday)
psun <- nrow(summer[summer$day == "Sunday" & is.na(summer$OCCUR_TIME),])/nrow(summer[summer$day == "Sunday",])
  # .0019

# Now for Winter
winter <- all_dates[all_dates$season == "Winter",]

# get P(no shooting Friday)
pf <- nrow(winter[winter$day == "Friday" & is.na(winter$OCCUR_TIME),])/nrow(winter[winter$day == "Friday",])
# .029

# get P(no shooting Saturday)
psat <- nrow(winter[winter$day == "Saturday" & is.na(winter$OCCUR_TIME),])/nrow(winter[winter$day == "Saturday",])
# .009

# get P(no shooting Sunday)
psun <- nrow(winter[winter$day == "Sunday" & is.na(winter$OCCUR_TIME),])/nrow(winter[winter$day == "Sunday",])
# .037

# And Now Spring
spring <- all_dates[all_dates$season == "Spring",]

# get P(no shooting Friday)
pf <- nrow(spring[spring$day == "Friday" & is.na(spring$OCCUR_TIME),])/nrow(spring[spring$day == "Friday",])
# .032

# get P(no shooting Saturday)
psat <- nrow(spring[spring$day == "Saturday" & is.na(spring$OCCUR_TIME),])/nrow(spring[spring$day == "Saturday",])
# .003

# get P(no shooting Sunday)
psun <- nrow(spring[spring$day == "Sunday" & is.na(spring$OCCUR_TIME),])/nrow(spring[spring$day == "Sunday",])
# .003
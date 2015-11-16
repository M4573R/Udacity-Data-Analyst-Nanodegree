## Summary

Taking advantage of Los Angeles open data, I took a deep dive analysis on the Los Angeles's violent crime rising in 2014 news report.  

While the overall number of violent crime rises in 2014 and so do the majority of districts, West LA, Pacific and Mission witness a drop in the number of violent crime in 2014 comparing with 2013.  
And despite of the overall fall in the number of property violent crime in 2014, property crime increases by 15% in Pacific District.  


## Background Story

I moved to LA for my new job as a data scientist a few months ago (Thanks Udacity! Bravo Data Analyst Nanodegree!). First thing first, I wanted to find a safe neighborhood to live in. So I started my research from [crimemapping.com](http://www.crimemapping.com) to LA news websites. Then I came across [this news on Los Angeles Times](http://www.latimes.com/local/crime/la-me-lapd-year-end-20141231-story.html) saying 'As 2014 draws to a close, the numbers show violent crime has climbed for the first time in 12 years.' I was planning to move to the West LA since my new company in the Santa Monica. I'm glad that data supports my decision -- West LA has relatively low number of both property and violent crimes and neither of them increase in 2014.  

## Design  

#### Time series lines?  
Right after I saw the news, I wanted to make multiple time series lines for each district for violent crimes. But due to the limitation of data and my interest of property crime number as well, I decided to do a year over year comparison between 2013 and 2014 in both property and violent crimes, just as what the news does, but in more details.  


#### Pie chart or (stack) chart bar?  
Multiple pie charts and (stack) bar charts were on top of my list for easy comparison between different categories.  

I wanted to focus my analysis only on violent and property crimes, while they take up only part of all the recorded crimes in LAPD provided data. Using a pie chart with only two categories would be misleading. So I decided to go with bar chart eventually.  


## Feedback 

I gathered three feedbacks from my coworkers and updated my data small multiple chart accorddingly.

From L: Put in a link of the news story you just show me! It'll lead your audience understand the background of the data visualization and conclude their own story better and easier.

From R: Try to switch to different colors and avoid red. Since your charts are related to crime, red kinda of hinting your andience that bar is more severe and deadly that others.

From J: You can use similar colors for the same category of crime for different year. This way, people would be easier to link them together and you can still keep them distinguishable.

## Resources

1. News on Los Angeles Times: http://www.latimes.com/local/crime/la-me-lapd-year-end-20141231-story.html  

2. Data Source: https://data.lacity.org/browse?q=LAPD%20Crime%20and%20Collision%20Raw%20Data&sortBy=relevance&utf8=%E2%9C%93

3. Jim Vallandingham's github repo: https://github.com/vlandham/co2_small_multiple 

4. CoffeeScript: http://coffeescript.org/
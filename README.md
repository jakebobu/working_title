# working_title
dsi project

## Concept

Predictive capability to grab future trends based on time series analysis of topics. Using natural language processing to pull out key/topic words from a news article, this app(?) will look to predict rising trends that have not reached the top of the pages in popularity. These predicted values will be determined by first and second order rates of growth in topics.

## Process

#### Start with having workflow given as a visualization

## Data

Depending on size may separate by source
Again if given enough articles may separate by provided topic/section
Otherwise one collective bin of all articles given a timestamp

### Detail sources, what news sources needed what, show how to reproduce data collection
* How I got keys, from whom
* requests library along with BeautifulSoup for we scraping
* For News API difference between description and full text

## Data Storage

nyt wash post, bbc news, breitbart, CNN, MSNBC

Look at WSJ

I am temporarily storing it on a MongoDB locally
Pete - move MongoDB credentials into environ

* Collective data is stored on S3 in following fashion:

| # | _id                  | headline              | pub_date   | section_name | web_url | word_count | content     | news_source |
| - | -------------------- | --------------------- | :--------: | ------------ |:-------:| :--------: | ----------- | ----------- |
| 0 | 09ig34w09ibs90iw34sb | Top 10 Reasons To...  | 0000-00-00 | NA           | https:  | 512        | This is a   | NYT         |
| 1 | tu8936nmvb09u8mtv4mu | Newsy News News...    | 0000-00-00 | Popular      | hhtps:  | 256        | story about | Wash Post   |
| 2 | tvs3um89psv48um9pet3 | Breaking News Here... | 0000-00-00 | Sports       | hhtps:  | 123        | how my life | ESPN        |

* id : unique identifier provided by source


## Data Manipulation

Test different methods of time series analysis <readings>
Try different methods of tokenizing/stemming/lemming spacy

Need to confirm consistency of column types across news sources (looking specifically at timestamps)

## Modeling

Rough numerical accelerated calculations
Topic selection is significant
Being able to generate new topics

Autoregressive–moving-average model
Autoregressive conditional heteroskedasticity

Holt-Winter Model (seasonality) exponentially weighted average

stats.models
pyflux.models -> ARIMA(X), GARCH, EGARCH, beta-t-EGARCH, GAS

Financial models 	(Read into more)
    Black–Derman–Toy Black–Karasinski Black–Scholes Chen Constant elasticity of variance (CEV) Cox–Ingersoll–Ross (CIR) Garman–Kohlhagen Heath–Jarrow–Morton (HJM) Heston Ho–Lee Hull–White LIBOR market Rendleman–Bartter SABR volatility Vašíček Wilkie


nmf

## Web App
Ran from AWS

### Live Data
Streaming data from news sources (decide upon update interval daily/weekly, create script to do that)
May refit on old predicted data, and predict on new data
Using time series analysis to pull rising trends
Currently thinking dashboard ran through flask unless I find a better alternative

### Visualization
Given model, show probability of new trending topics
* Word clouds
* Plots across time

## Deliverables
* Model that can grab current data over a prior period and predict trendingnessity(tm)
* Web App to present that data (probably through flask on EC2 instance)
* Clean and efficient storage of model and data through S3 and on EC2 instance


## Schedy

11/14-11/17
* convert from json dump into an organized fashion containing only important features
* create a pipeline for natural language processing where I can test various methods
* create a pipeline for time series analysis where "                  "

11/18-11/19
* Get an MVP model ready to go
* Test various methodologies and rate their general performance

11/20-11/24
* Make final selection on model and look to optimize parameters
* Quantify time series analysis (what growth rates are we looking for?)
* Setup MVP Web App that just gets it done
* Look into public/private access rights on AWS EC2 and S3 to makes sure another can use it (but wait till ready)

11/25-end
* Make web app user friendly and easy on the eyes
* Let it run for test periods of times by itself (I'm not logged in)
* Open to public access (maybe?)
* Finish this markdown and make it look good



## Future Goals
Everything I can't do in time
Chronos Job as future

To Read:
https://en.wikipedia.org/wiki/Stochastic_simulation
https://en.wikipedia.org/wiki/Time_series
https://en.wikipedia.org/wiki/Autoregressive%E2%80%93moving-average_model


Things I did today (11/15)
Converted mvp pipeline into a class object
Add verbosity into class object
Did some modifications on spacy toky/lemy
Generate counts of each article in terms of topics by specified dt
Did rough calc to get vel and accel, shows times where v > 0 and a > 0

TODO tomorrow (11/16)
Look at what I got, see if it makes sense (tokenizer needs more work. 1 topic is phone numbers, emails contact us at) taxes -> taxis
Improve accel algorithm, look for algorithms that utilize acceleration and are powerful in prediction
Have a feeling that some newfs sources articles are too sparse (across time creating) creating topics by source itself
Consolidate all variables I can test on in one location (make easier for myself later when playing with them)
Look to set an initial threshold or comparison metric for acceleration. See what I am seeing rising and when, again based on accel alogithm
Pickle problem, why can't I pickle my model











*** END  ***

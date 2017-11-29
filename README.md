# Time Series Trending of News Articles Organized through Latent Topics
Galvanize DSI Capstone

## Table Of Contents

* [Concept](#concept)
* [Process](#process)
* [Data](#data)
  * [Web Scrounging and Scraping](#web-scrounging-and-scraping)
  * [Data Storage](#data-storage)
  * [Data Manipulation](#data-manipulation)
* [Modeling](#modeling)
  * [Hyper-Parameter Selection](#hyper-parameter-selection)
  * [Predictive Algorithms](#numerical-predictive-forecasting-algorithm)
* [Web App](#web-application)
  * [Functionality](#functionality)
  * [AWS EC2 Instance](#aws-ec2-instance)
* [Future Goals](#future-goals)
  * [Live Data](#live-data)
  * [News Source Analysis](#news-source-analysis)
* [Acknowledgments](#acknowledgments)

## Concept

The purpose of this project is to generate a model with the predictive capability to grab future trends based on time series analysis of topics. Using natural language processing and Non-negative Matrix Factorization (NMF) to pull out key/topic words from news articles, this model looks to see latent topics that have rising rates of publication. These trends and topics will then be displayed through a web application.

## Process

### Workflow Visualization

  ![Flow Chart of Process](readme_images/workflow.png "Flow Chart of the workflow")

## Data

News Articles from all topics/sections over a period of at the time previous 5 months (July-2017 to November-2017)

### Web Scrounging and Scraping

Listing of articles were either provided through an API interface (NYT API website, NEWS API website), or through web scraping news sources' article archive search. Requests library from python along with BeautifulSoup allow for web scraping and extracting the important elements of the articles. This is ran through scrape_web.py.

### Data Storage

After data is collected, it is stored in a .csv on AWS S3. The following things are included about the articles: a unique identifier provided by source if one exists, the article headline, the article publication date, the section the article is found under (e.g. Arts, Politics, Sports), the URL to that article, the count of words in an article (used to eliminate extremely short articles), and the name of the news source. Implementation on pushing and pulling of data on S3 is done in post_to_s3.py.


### Data Manipulation

The first goal is to convert article text into a list of words (tokenizing) and group similar words, especially synonyms, into a singular root word (lemmatizing). The spaCy library accomplishes this task. Additionally, tokens containing only punctuation or numbers are removed from the list as they do not add to the specific content of an article. Once an article text is converted to its list of words, it is vectorized through sklearn's TfidfVectorizer in order to emphasize the unique words in each article. This vectorized corpus is then ran through sklearn's NMF model to generate latent topics. These steps are performed by the class NMF_Time that is found /flask_app/words_to_vals.py.

  ![Article Text to Tokens](readme_images/text_to_tokens.png "Converting article text to list of tokens")

  <br>

## Modeling

### Hyper-parameter Selection

Deciding upon the number of topics to have is significant to the outcome of the predictive modeling. Too few topics means they are too generalized, and too many topics means that there would not be enough articles in each topic to conduct time series analysis on. Selection of number of topics is done by comparing the cosine similarity between topics, cosine similarity between articles in a topic, and the reconstruction error for the Non-Negative Matrix Factorization. Varying from 10 to 1000 number of topics, a final number of 500 topics was selected. Above this point, there was minimal improvement upon similarities and error.

A threshold of a minimum cosine similarity between articles and topics determines at what point an article becomes related to, or part of, a topic based on that similarity. An optimal threshold is one that is small enough to have the greatest percentage of articles have a similarity to a topic greater than that threshold while not too small and having articles related to too many topics. Comparing a range of thresholds shows an optimal threshold of ~0.05.

![Threshold Selection](readme_images/article_threshold.png "Threshold Selection for Article to Topic Similarity")

<br>


### Numerical Predictive Forecasting Algorithm
Provided a listing of articles and their similarity to topics, the articles are separated into related topics and by date and time of publication. This is done as a following step to above in the class NMF_Time found /flask_app/words_to_vals.py. These counts then need to be smoothed and ran through a preditive model to forecast future counts.

#### Holt-Winter Model (seasonality) exponentially weighted average
This model utilizes triple exponential forecasting to predict future trends. The reason for this model's selection is due to the seasonality that is noticed for topics on a weekly (7-day) cycle. This is particularly true for topics containing articles published on the weekend. Holt-Winter Model incorporates three hyper-parameters:
  * Alpha - the importance of the prior values
  * Beta - the importance of the trending behavior (rate of change)
  * Gamma - the importance of seasonality

The optimal values for these parameters are found using scipy's optimize library. The work for this step is seen under flask_app/find_best_hyperparams.py. The implementation of this model is found in the Count_Worker clas in flask_app/work.py.

<br>

![Double Exponential Hyper-parameters](readme_images/double_exp_param_search.png "Optimal Double Exponential Hyper-Parameters of alpha=0.37, beta=1.0")

Sum of Squared Errors across a subset of the article corpus to show ranges of errors across ranging Hyper-Parameters (Black point indicates location of minimum error)
<br>

## Web Application
Ran from AWS utilizing flask through python. See flask_app/app.py for its code.
Visit Website Here:
(If down, contact me for assistance)
<!-- ![Flask App Index Page](readme_images/index_page.png "Landing Page of Flask Web Application") -->


### Functionality
Current functionality of the web application includes the following:

#### Word Clouds of the top words per topic.
Given a provided topic, presents the words that make up that topic sized by the relative importance of each word in that topic.
<!-- ![Flask App Time Trend](readme_images/word_cloud.png "Example Word Cloud from Flask Web Application") -->

#### Plots across time
Shows a recent time trend of article counts from a provided topic, including a prediction on future behavior.
<!-- ![Flask App Word Cloud](readme_images/time_trend.png "Example Time Trend from Flask Web Application") -->

#### Articles from Specific Topic
Given a provided topic, lists the articles that constitute that topic.
<!-- ![Flask App Word Cloud](readme_images/article_listing.png "Example Listing of Articles related to a Topic from Flask Web Application") -->

#### Topics from Specific Word
Given a provided token (word), lists the topics that contain that token.
<!-- ![Flask App Word Cloud](readme_images/topic_listing.png "Example Listing of Topics related to a Word from Flask Web Application") -->

### AWS EC2 Instance
* Allows flask web application to be continuously ran and accessed from any location
* General instructions to setup instance (depends on how user wants to provide data):
  * AWS setup -> Specify IAM role to get data from S3 bucket if that is source of articles
  * EC2 Instance requires standard python/anaconda suite of libraries, but additionally needs boto3 if S3 utilized, spaCy for natural language processing, and flask to run web application

## Future Goals

### Live Data
* Automated streaming data from news sources (decided upon update interval daily/weekly)
* Automated retraining of model and topic generation


### News Source Analysis
* Provided a greater and more comprehensive corpus of news articles, train independently on news source to see differences amongst news sources
* Similarity, train independently on provided news section (Sports, Arts, World) to see impact on time series analysis


## Acknowledgments
* spaCy - https://spaCy.io/
* jQCloud - http://mistic100.github.io/jQCloud/index.html
* Special thanks to NYT and NEWS API for providing easy access to article collections
* Grisha Trubetskoy's Holt-Winter Blog for resource on implementation of said model - https://grisha.org/blog/2016/01/29/triple-exponential-smoothing-forecasting/



<!-- AWS setup -> IAM role to get data from S3 bucket
EC2 Instance -> install boto3: pip install boto3
                    b3c = boto3.resource('s3')
                    bucket = b3c.Bucket('peterrussodsiproj')
                    <In repo root directory>
                    bucket.download_file('temp_data1.csv','temp_data1.csv')
                install spaCy: conda install -c conda-forge spaCy
                               python -m spaCy download en
                install pyflux: pip install pyflux

ctrl+b release then d



## Schedy (for me)

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


train on topics with over a threshold of counts

Thursday Presentations 4 minutes
Monday dress rehersal

*** END  *** -->

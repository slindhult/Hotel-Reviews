## Predicting Hotel Ratings from Hotel Reviews
![Header Image](https://cache.marriott.com/marriottassets/destinations/hero/europe-destination.jpg?interpolation=progressive-bilinear)

### Table of Contents
<ul>
<li><a href="https://github.com/slindhult/Hotel-Reviews/#Hotel-Review-Data">Data Exploration</a></li>
<li><a href="https://github.com/slindhult/Hotel-Reviews/#Mapping">Geographical Data Visualization,</a></li>
<li><a href="https://github.com/slindhult/Hotel-Reviews/#Cleaning-and-Feature-Engineering">Cleaning</a>,</li>
<li><a href="https://github.com/slindhult/Hotel-Reviews/#Modeling">Modeling</a></li>
<li><a href="https://github.com/slindhult/Hotel-Reviews/#Next-Steps">Next Steps</a></li>
</ul>


### **Hotel Review Data:**
<hr/>
The dataset contained 515,000 Hotel Reviews pulled from Booking.com.  The reviews were rated on a 1-10 scale and drawn from 5 major European cities: Amsterdam, Barcelona, London, Paris and Vienna. The dataset included a positive and negative review column, as well as average hotel rating, number of reviews, reviewer nationality and basic information on their stay.  The goal of this project was to build a model to predict the reviewer score based on the data available and the sentiment analysis of their review.
The dataset used was retrieved from: 

[Kaggle](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)


A quick description of the dataset shows that the mean is 8.39 and the standard deviation is 1.63, this means all reviews above the mean are within one standard deviation and those towards the lower end (minimum rating of 2.5) are further away.

![Image of dataset description](https://github.com/slindhult/Hotel-Reviews/blob/master/images/Screenshot%20from%202020-05-15%2008-24-53.png?raw=true)


![Distribution of Reviews](https://github.com/slindhult/Hotel-Reviews/blob/master/images/%20hist.png?raw=true)



### Mapping
<hr />
To better visualize the hotel locations and ratings,  an interactive map was created plotting the hotels using their latitude, longitude and star ratings broken out by quartile.  Below is snapshot of central Paris:

![Paris Hotel Map](https://github.com/slindhult/Hotel-Reviews/blob/master/images/paris.png?raw=true)


### **Cleaning and Feature Engineering:**
<hr />
Datapoints without any reviews were dropped, as the focus was on natural language processing.  The tags column contained many unique room descriptions, so feature engineering was completed to categorize rooms as either budget, medium, high, or fancy.  This allowed the data to be used and greatly reduce the number of features that would have been required.  Additional information about their travel was also included: number of nights stayed, couple, family, solo, and if they were traveling for business.  Vader Sentiment Analysis and TextBlob sentiment analysis were used to generate reviewer sentiment scores for both positive and negative reviews.  Stopwords were removed and keywords were lemmatized prior to adding a count vectorizer matrix based on the top 300 words from both positive and negative reviews.



Non-negative Matrix factorization was used to identify latent topics.  There were three major human identifiable topics that stoood out from the positive reviews:

### &emsp;&emsp;&emsp;&emsp;      Hotel &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;                          Room     &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Location  

<p float="left">
  <img src="https://github.com/slindhult/Hotel-Reviews/blob/master/images/Topic1.png?raw=true" width="275" />
  <img src="https://github.com/slindhult/Hotel-Reviews/blob/master/images/topic2.png?raw=true" width="275" /> 
  <img src="https://github.com/slindhult/Hotel-Reviews/blob/master/images/topic3.png?raw=true" width="275" />
</p>

To understand the most significant words, counts of words in positive and negative reviews were combined to differentiate which words showed up more in positive and negative reviews, values show how many more times it showed up in the respective reviews category:


<p float="middle">
  <img src="https://github.com/slindhult/Hotel-Reviews/blob/master/images/poswords.png?raw=true" width='415' />
  <img src="https://github.com/slindhult/Hotel-Reviews/blob/master/images/negwords.png?raw=true" width="415" /> 
</p>

The best review according to Vader Sentiment Analysis:
>' Upon check in the staff were all so polite and welcoming to stepping through the doors We were served by Shady who really polite and friendly and let us know about the other amazing little perks about staying here to the love music on the evening and spa treatments also Always had bad experiences with hotels just but staying here has just been above and beyond anything we stayed the 1 night and I ve never been amazed as soon as I walked into our hotel room as I did here The room was spotless and bathroom was fantastic with a separate shower and a superb bath which I loved Everything is so easy to get too and within walking distance from either Embankment or Waterloo station couldn t fault anything Celebrating my boyfriends birthday before he is deployed for a while this was our last little trip together till next year and arriving back in our hotel there was a lovely little note and an AMAZING CHEESECAKE left for us reading Happy Birthday and to enjoy our stay from Celeste Sullivan Guest relations manager I was certainly not expecting anything so kind and for that I m beyond grateful Breakfast in the morning was just brilliant the staff were soo efficient and quick and still so friendly the choice of food was brilliant even for the picky eaters like my self and winner of Peanut Butter Checking out was a breeze and we were then served by the lovely Nicole who made sure our stay went well and was really friendly the kind of service you expect to get from any hotels you tend to stay at and I will never stay anywhere other than here in the future Thank you Park Plaza for the most amazing experience here at your wonderful hotel me and my partner were amazed by your service and can t wait to come back again next year HIGHLY RECOMMENDED and family friendly too you won t be disappointed x X x '


The worst review according to Vader Sentiment Analysis:
>'So so so bad experience and memories when I was in the hotel The first night when I arrive in the hotel after a long long journey so so tired but when I want to take a shower before sleep the water was totally cold cold totally cold no hot water Then I call the reception knowing that the system of the whole hotel was wrong Ok I can understand the system broken but why do not inform the clients early The hotel know it before but did not tell the client I have no choose and then very sad and tired without a shower and then I pay cash to the opposite another hotel to rent a room for shower in the first night I arrive in paris alone and midnight And what is more terrible is the second day morning when I told the new reception what happened last night the reception named Jimmy with very bad attitude told me ok we have problem with the system but we have already make it well now you have already the hot water And I said so how about last night You even did not say sorry and I even pay cash to another hotel for a shower in the midnight and Jimmy said so ok what do you want last night is past and past is past You need to pay So show me the prove Oh my god I even do not believe my ear i stand there do not believe what he said I never let the hotel to pay the fees for shower but my god they even did not say sorry in the second day So rude And make me really angry I travelled all over the world and it was my sixth time in paris but the first time to meet so ugly hart and bad rude guy Even the reception of the first night boy was so sorry and said so shamed for no hot water with many polite but the old reception Jimmy are so so so bad polite'

### Modeling
<hr />
The goal was to create a model to predict the user's review score based on their positive and negative reviews.  To create a baseline to measure my models against, the hotel's average rating was predicted for each datapoint and the mean absolute error was taken on the results. This gave a baseline to beat of 1.18 stars.
The next step was to use sentiment analysis on the reviews.  Both Vader sentiment analysis from NLTK and TextBlob sentiment analysis were run on the data set for comparison.  Vader performed slightly better and was used moving forward.  Next stopwords were removed and the top 300 words from each type of review were added as features to the dataset as a term frequency matrix.  The vader sentiment lexicon was originally trained on social media data, to better tailor it to the hotel dataset some of the most frequently occuring words were added to the lexicon to more accurately predict sentiment.


Below is a visualization of how each model performed on the various datasets:
![Image of Model Performance](https://github.com/slindhult/Hotel-Reviews/blob/master/images/modelcomparison.png?raw=true)


The best performance was from a gradient booosted model run on the vaderanalysis dataset with term frequency matrix of the top 300 words, achieving a mean absolute error of 0.77 stars.  The model was tuned using randomsearch to find the best parameters given the run time for each iteration. The top 15 feature importances (_neg indicates that it is from the countvectorizer of the negative reviews) were pulled from the best model:
1) room_neg, 3.213848362698663
2) new_neg_sent, 1.3588873831895074
3) new_pos_sent, 1.197688207777768
4) staff_neg, 1.1175715739400547
5) dirty_neg, 0.9736228654579733
6) fault_neg, 0.7786412114249133
7) perfect_neg, 0.7507870704583097
8) cleanliness, 0.702056108507394
9) money_neg, 0.6654582424585267
10) star_neg, 0.6624058946175011
11) Review_Total_Negative_Word_Counts, 0.643662427137203
12) rooms_neg, 0.6394214532205174
13) clean_neg, 0.6272994880442819
14) hotel_neg, 0.6126645258793607
15) staff, 0.6016395407762328

Many of the top features come from the negative review list.  This is likely because many of the ratings are grouped around the mean at 8.4, and the model needs to account for the lower scores.

### Next Steps
<hr />

* The term frequency matrix size was limited on my local machine, given the opportunity adding terms may yield better results.
* Testing the model on data from other booking websites or regions to gauge the generalizability would be interesting.

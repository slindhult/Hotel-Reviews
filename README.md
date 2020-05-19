## Hotel-Reviews
![Image of Emissions](https://cache.marriott.com/marriottassets/destinations/hero/europe-destination.jpg?interpolation=progressive-bilinear)
### **Data:**
The dataset contained 515,000 Hotel Reviews pulled from Booking.com.  The reviews were rated on a 1-10 scale and drawn from 5 major European cities: Amsterdam, Barcelona, London, Paris and Vienna. The dataset included a positive and negative review column as well as average hotel rating, number of reviews, reviewer nationality and some information basic information on their stay.  The goal of this project was to build a model to predict the reviewer score based on the data available as well as sentiment analysis of their review.
The dataset used was retrieved from kaggle: https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe


Getting a quick description of the datset shows that the mean is 8.4 and the standard deviation is 1.6, this means all reviews above the mean are within one standard deviation and those towards the lower end (min rating of 2.5) are quite far away.

![Image of dataset description](https://github.com/slindhult/Hotel-Reviews/blob/master/images/Screenshot%20from%202020-05-15%2008-24-53.png?raw=true)


![Image of Emissions](https://github.com/slindhult/Hotel-Reviews/blob/master/images/%20hist.png?raw=true)



### Mapping
To better visualize the hotel locations and ratings,  a map was created plotting the hotels using their latitude, longitude and star ratings colored by quartile.
![Image of Emissions](https://github.com/slindhult/Hotel-Reviews/blob/master/images/Screenshot%20from%202020-05-16%2022-34-44.png?raw=true)


### **Cleaning and Feature Engineering:**
Datapoints without any reviews were dropped as the focus was on natural language processing.  The tags column contained many unique room descriptions so feature engineering was completed to determine the type of room/hotel they stayed in as either budget, medium, premium, or fancy.  Additional information about their travel was also included: number of nights stayed, couple, family, solo, and if they were traveling for business.  Vader Sentiment Analysis and textblob sentiment analysis were used to generate sentiment analysis scores for both the positive and negative reviews.  Stopwords were removed and words were lemmatized before adding a term frequency matrix based on the top 500 words from each positive and negative reviews.



Non-negative Matrix factorization was used to identify latent topics, there were three major human identifiable topics that stoood out from the positive reviews:

### &emsp;&emsp;&emsp;&emsp;      Hotel &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;                          Room     &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Location  

<p float="left">
  <img src="https://github.com/slindhult/Hotel-Reviews/blob/master/images/Topic1.png?raw=true" width="275" />
  <img src="https://github.com/slindhult/Hotel-Reviews/blob/master/images/topic2.png?raw=true" width="275" /> 
  <img src="https://github.com/slindhult/Hotel-Reviews/blob/master/images/topic3.png?raw=true" width="275" />
</p>

To understand the most significant words, words for positive and negative reviews were combined to differentiate which words showed up more in positive and negative reviews, values show how many more times it showed up in the respective reviews category:


<p float="middle">
  <img src="https://github.com/slindhult/Hotel-Reviews/blob/master/images/poswords.jpg?raw=true" width='425' />
  <img src="https://github.com/slindhult/Hotel-Reviews/blob/master/images/negwords.jpg?raw=true" width="425" /> 
</p>

### Modeling
I wanted to create a model that would predict the user's review score based on their positive and negative reviews.  To set a base line I took the mean absolute error of predicting the hotel's average rating every time, this gave a baseline of 1.18 stars.
The next step was to use sentiment analysis of the reviews, I compared the vader sentiment analysis library from NLTK and the TextBlob sentiment analysis.  Vader performed slightly better so I stuck with that.  Next stopwords were removed and the top 300 words from each type of review were added as features to the dataset using CountVectorizer.  The vader sentiment lexicon (positive and negative association for certain words ranging from -4 to 4) was slightly adjusted to better fit the hotel dataset.


Below is a visualization of how the models performed on the datasets:
![Image of Model Performance](https://github.com/slindhult/Hotel-Reviews/blob/master/images/model-comparison.png?raw=true)


The best model performance was achieved with XGBoost on the dataset with vaderanalysis and CountVectorizer of the top 300 words, achieving a mean absolute error of 0.77 stars.

#### Top 15 feature importances (_neg indicates that it is from the countvectorizer of the negative reviews)
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

Many of the top features come from the negative review list, this is likely because many of the ratings are grouped around the mean at 8.4 and the model needs to account for the lower scores which are farther away.

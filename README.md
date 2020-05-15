## Hotel-Reviews
![Image of Emissions](https://cache.marriott.com/marriottassets/destinations/hero/europe-destination.jpg?interpolation=progressive-bilinear)
### **Data:**
The dataset contained 515,000 Hotel Reviews pulled from Booking.com.  The reviews were rated on a 1-10 scale and drawn from 5 major European cities: Amsterdam, Barcelona, London, Paris and Vienna. The dataset included a positive and negative review column as well as average hotel rating, number of reviews, reviewer nationality and some information basic information on their stay.  The goal of this project was to build a model to predict the reviewer score based on the data available as well as sentiment analysis of their review.

The dataset used was retrieved from kaggle: https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe




### **Cleaning and Feature Engineering:**
Datapoints without any reviews were dropped as the focus was on natural language processing.  The tags column contained many unique room descriptions so feature engineering was completed to determine the type of room/hotel they stayed in as either budget, medium, premium, or fancy.  Additional information about their travel was also included: number of nights stayed, couple, family, solo, and if they were traveling for business.  Vader Sentiment Analysis and textblob sentiment analysis were used to generate sentiment analysis scores for both the positive and negative reviews.  Stopwords were removed and words were lemmatized before adding a term frequency matrix based on the top 500 words from each positive and negative reviews.




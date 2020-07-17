url = input('Insert the booking.com url of the hotel you would like to scrape:')



import pandas as pd
from bs4 import BeautifulSoup
from urllib.request import urlopen as uReq
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
import re
import json
import pickle


with open('positive_vocab.txt', 'r') as filehandle:
    poslist = json.load(filehandle)
with open('negative_vocab.txt', 'r') as neg:
    negative = json.load(neg)

def make_new_data(url):
    uClient = uReq(url)
    page_html = uClient.read()
    page_soup = BeautifulSoup(page_html, 'html.parser')
    reviews = page_soup.find_all(class_="review_item")
    avg = str(page_soup.find(class_="bui-review-score__badge"))
    average_score = float(re.findall('Scored(.+)" class',avg)[0])
    review_list = []
    for review in reviews:
        string = str(review)
        reviewer_score = float(re.findall('Scored (.+) "', string)[0])

        tag = re.findall('•(.+)', string)
        tags = [i[8:] for i in tag]
        try:
            neg = string.split('</svg>')[1].split('</p>')[0]
        except:
            neg = 'No Negative'
        try:
            pos = string.split('</svg>')[2].split('</p>')[0]
        except:
            pos = 'No Positive'
        review_dict = {'Average_Score':average_score, 'Reviewer_Score':reviewer_score, 'Tags':tags,
                      'Negative_Review':neg, 'Positive_Review':pos}
        review_list.append(review_dict)
    return review_list

def clean_tags(dataset, column_name):
    for index, i in enumerate(dataset[column_name]):
        for idx, t in enumerate(i):
            if t not in ['Stayed 1-2 nights','Stayed 3-4 nights', 'Stayed 5+ nights', 'Business trip', 'Solo traveler', 'Leisure trip',
                         'Couple', 'Group', 'Family with young children', 'Family with older children']:
                if t in ['Stayed 1 night','Stayed 2 nights']:
                    dataset[column_name][index][idx] = 'Stayed 1-2 nights'
                if t in ['Stayed 3 nights','Stayed 4 nights']:
                    dataset[column_name][index][idx] = 'Stayed 3-4 nights'
                if t in ['Stayed 5 nights','Stayed 6 nights', 'Stayed 7 nights', 'Stayed 8 nights', 'Stayed 9 nights',
                         'Stayed 10 nights',  'Stayed 11 nights',
                     'Stayed 12 nights', 'Stayed 13 nights', 'Stayed 14 nights', 'Stayed 15 nights', 'Stayed 16 nights',
                         'Stayed 17 nights','Stayed 18 nights', 'Stayed 19 nights', 'Stayed 20 nights',
                     'Stayed 21 nights', 'Stayed 22 nights', 'Stayed 23 nights', 'Stayed 24 nights', 'Stayed 25 nights',
                         'Stayed 26 nights',
                     'Stayed 27 nights', 'Stayed 28 nights', 'Stayed 29 nights', 'Stayed 30 nights', 'Stayed 31 nights',]:
                    dataset[column_name][index][idx] = 'Stayed 5+ nights'
    for i in dataset[column_name]:
        for idx, t in enumerate(i):
                if t not in ['Stayed 1-2 nights','Stayed 3-4 nights', 'Stayed 5+ nights', 'Business trip',
                             'Solo traveler', 'Leisure trip', 'Couple', 'Group', 'Family with young children',
                             'Family with older children']:
                    i.pop(idx)
    from sklearn.preprocessing import MultiLabelBinarizer
    mlb = MultiLabelBinarizer()

    tagdf = pd.DataFrame(mlb.fit_transform(dataset[column_name]),columns=mlb.classes_, index=dataset.index)


    dataset = dataset.join(tagdf)
    dataset.drop(column_name, axis=1, inplace=True)

    return dataset

def get_vader(dataframe, negative_review_col, positive_review_col):

    analyser = SentimentIntensityAnalyzer()
    dataframe[negative_review_col] = dataframe[negative_review_col].apply(lambda x: str(x).replace("No Negative", ""))
    dataframe[positive_review_col] = dataframe[positive_review_col].apply(lambda x: str(x).replace("No Positive", ""))

    dataframe['vader_pos_sent'] = dataframe[positive_review_col].apply(lambda x: analyser.polarity_scores(x)['compound'])
    dataframe['vader_neg_sent'] = dataframe[negative_review_col].apply(lambda x: analyser.polarity_scores(x)['compound'])

def review_word_count(dataframe, negative_review_col, positive_review_col):

    analyser = SentimentIntensityAnalyzer()

    dataframe['Review_Total_Negative_Word_Counts'] = dataframe[positive_review_col].apply(lambda x: len(x.split()))
    dataframe['Review_Total_Positive_Word_Counts'] = dataframe[negative_review_col].apply(lambda x: len(x.split()))


def count_vectorize(dataframe, negative_review_col, positive_review_col, poslist, neglist):
    countpos = CountVectorizer(stop_words='english', vocabulary = poslist)
    countneg = CountVectorizer(stop_words='english', vocabulary = neglist)

    pos_count = countpos.fit_transform(dataframe[positive_review_col].values.astype('U'))
    pos_col_names = countpos.get_feature_names()
    pos_count = pos_count.todense()
    pos_count = pd.DataFrame(pos_count, columns = pos_col_names)
    dataframe = dataframe.join(pos_count)

    neg_count = countneg.fit_transform(dataframe[negative_review_col].values.astype('U'))
    neg_col_names = countneg.get_feature_names()
    neg_count = neg_count.todense()
    neg_count = pd.DataFrame(neg_count, columns = neg_col_names)
    neg_count = neg_count.add_suffix('_neg')
    dataframe = dataframe.join(neg_count)

    dataframe.drop(negative_review_col, axis=1, inplace=True)
    dataframe.drop(positive_review_col, axis=1, inplace=True)
    return dataframe


def fill_missing_cols(dataframe, original_dataframe_cols):
    missing = [i for i in original_dataframe_cols if i not in dataframe.columns]
    for i in missing:
        dataframe[i] = 0
    return dataframe


hotel = make_new_data(url)
data = pd.DataFrame(hotel)


general = pd.read_csv('generalizednew.csv')

data = clean_tags(data, 'Tags')
get_vader(data, 'Negative_Review', 'Positive_Review')
review_word_count(data, 'Negative_Review', 'Positive_Review')
data = count_vectorize(data, 'Negative_Review', 'Positive_Review', poslist, negative)
fill_missing_cols(data, general.columns)

new = pd.concat([general, data],sort=True)
new.to_csv('generalizednew.csv')
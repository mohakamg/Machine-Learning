import tweepy
from textblob import TextBlob

consumer_key = 'wAqJbKdKtlXACCPfTNGaMBcIF'
consumer_secret = 'ZMK80onHeeoWkVXm2OPHtsZzQdn9cIH0mFfr8IynFWEyS7cLDl'

access_token = '822567980257177600-QdhGVEXa19yATorTstb4lrKGRMkeAdQ'
access_token_secret = 'hG7vogmbvJn1iRTGWCgMmucy5MUkXoXCSrJPB7GPeIePc'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

public_tweets = api.search('Real Estate Speculation')

for tweet in public_tweets:
    print(tweet.text)
    analysis = TextBlob(tweet.text)
    print(analysis.sentiment)

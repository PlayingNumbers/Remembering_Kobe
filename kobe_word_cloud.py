# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 12:14:29 2020

@author: Ken
"""

from twitterscraper import query_tweets 
import datetime as dt 
import pandas as pd 
from wordcloud import WordCloud, ImageColorGenerator, STOPWORDS
import matplotlib.pyplot as plt 
from PIL import Image
from os import path, getcwd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

#picture from https://www.kindpng.com/imgv/wRiixm_kobe-bryant-los-angeles-lakers-nba-jersey-detroit/

begin_date = dt.date(2020,1,26)

tweets = query_tweets("#Kobe", begindate = begin_date)
                      
df = pd.DataFrame(t.__dict__ for t in tweets)
df.to_csv('kobe_tweets.csv', index = False)

kobe_data = pd.read_csv('kobe_tweets.csv')
words = " ".join(kobe_data.text.drop_duplicates())

def punctuation_stop(text):
    """remove punctuation and stop words"""
    filtered = []
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    for w in word_tokens:
        if w not in stop_words and w.isalpha():
            filtered.append(w.lower())
    return filtered


words_filtered = punctuation_stop(words)

#get the working directory 
d = getcwd()

unwanted = ['kobe','bryant','kobebryant','https','helicopter', 'crash','today','rip']

#remove unwanted words 
text = " ".join([ele for ele in words_filtered if ele not in unwanted])

#numpy image file of mask image 
mask_logo = np.array(Image.open(path.join(d, "kobe_image_white.png")))

#create the word cloud object 
wc= WordCloud(background_color="white", random_state=1, mask=mask_logo, stopwords=STOPWORDS, max_words = 2000, width =800, height = 1500)
wc.generate(text)

#wc= WordCloud(background_color="rgba(255, 255, 255, 0)", mode="RGBA", random_state=1, mask=mask_logo, stopwords=STOPWORDS, max_words = 2000, width =800, height = 1500)
#wc.generate(text)

image_colors = ImageColorGenerator(mask_logo)
wc.recolor(color_func=image_colors).to_file('kobe_cloud.png')

plt.figure(figsize=[10,10])
plt.imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")
plt.axis('off')
plt.show()

wc.tofile()


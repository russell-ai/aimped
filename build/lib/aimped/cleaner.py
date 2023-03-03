
import re
# import nltk
# from nltk.corpus import stopwords


def text_cleaner(text):
    # lower case
    text = text.lower()
    # remove punctuation
    text = re.sub(r'[^\w\s]','',text)
    # remove numbers
    text = re.sub(r'\d+','',text)
    # remove stop words
    # text = ' '.join([word for word in text.split() if word not in stop_words])
    # remove short words
    text = ' '.join([word for word in text.split() if len(word) > 1])
    # remove non-ascii characters
    text = text.encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

if __name__ == "__main__":
    text = "This is a test°′﻿. This is only a test."
    print(text_cleaner(text))
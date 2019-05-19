# %%
import pandas as pd
import re
from textblob import TextBlob
from nltk.tokenize import TweetTokenizer
from collections import defaultdict

def openTSV(filepath: str):
    return pd.read_csv(filepath, sep='\t')


def extractKeywords(tweet: str):
    tw = TweetTokenizer(strip_handles=True, reduce_len=True)

    # remove all emoji
    tweet = re.sub(r'\\[a-z0-9]{5}', '', tweet)

    # remove all links
    tweet = re.sub(r'(http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/)?[a-z0-9]+([\-\.]{1}[a-z0-9]+)*\.[a-z]{2,5}(:[0-9]{1,5})?(\/.*)?', '', tweet)
    
    # extract and then remove hashtags
    hashtags = re.findall(r"#\w+", tweet)
    tweet = re.sub(r'#\w+', '', tweet)
    tweet = ' '.join(tw.tokenize(tweet))
    blob = TextBlob(tweet)
    keywords = list(blob.noun_phrases) + hashtags
    return keywords

#%%
def preprocess(trainFilePath: str, storeFilePath: str="C:\\Users\\novan\\OneDrive\\Desktop\\CODE\\twitter-geotagging\\data\\preprocessed.csv"):
    st = open(storeFilePath, 'w')
    df = openTSV(trainFilePath)
    freq = defaultdict(lambda: defaultdict(int))
    classes = []
    for i, row in df.iterrows():
        keywords = extractKeywords(row[2])
        classes.append(row[1])
        for k in keywords:
            freq[k][i] += 1

    attributes = sorted(freq.keys())
    header = 'ID,Location,' + ','.join(attributes) + '\n'
    st.write(header)

    for i in range(len(classes)):
        values = str(freq[attributes[0]][i])
        for a in range(1, len(attributes)):
            values += ',{}'.format(freq[attributes[a]][i])
        st.write("{},{},{}\n".format(i+1, classes[i], values))
            
    st.close()
    return

# %%
filepath = "C:\\Users\\novan\\OneDrive\\Desktop\\CODE\\twitter-geotagging\\data\\train-raw.tsv"
preprocess(filepath)
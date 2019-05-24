# %%
import pandas as pd
import re
from textblob import TextBlob, Word
from nltk.tokenize import TweetTokenizer
from collections import defaultdict
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords

def openTSV(filepath: str):
    return pd.read_csv(filepath, sep='\t')


def extractKeywords(tweet: str):
    stops = stopwords.words('english')
    tw = TweetTokenizer(strip_handles=False, reduce_len=True)

    # remove all emoji
    tweet = re.sub(r'\\[a-z0-9]{5}', '', tweet)

    # remove all links
    tweet = re.sub(r'(http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/)?[a-z0-9]+([\-\.]{1}[a-z0-9]+)*\.[a-z]{2,5}(:[0-9]{1,5})?(\/.*)?', '', tweet)

    # extract and then remove hashtags
    hashtags = re.findall(r"#\w+", tweet)
    tweet = re.sub(r'#\w*', '', tweet)
    tweet = ' '.join(tw.tokenize(tweet))
    blob = TextBlob(tweet)
    nouns = [word for word in blob.noun_phrases if word not in stops]
    nouns = [Word(word) for word in nouns]
    nouns = [word.strip(' /\\*.;,') for word in nouns if word.strip(' /\\*.;,') != '']
    keywords = nouns + hashtags
    return keywords

#%%

def preprocess(trainFilePath: str, storeFilePath: str="C:\\Users\\novan\\OneDrive\\Desktop\\CODE\\twitter-geotagging\\data\\preprocessed3.csv"):
    st = open(storeFilePath, 'w')
    df = openTSV(trainFilePath)
    # freq = defaultdict(lambda: defaultdict(int))
    # classes = []
    st.write('ID\tLocation\tkeywords\n')
    for i, row in df.iterrows():
        keywords = extractKeywords(row[2])
        if keywords == []:
            continue
        st.write("{}\t{}\t{}\n".format(i+1, row[1], ','.join(keywords)))

            
    st.close()
    return


def deleteUnnecessaryWord(h):
    x = defaultdict(lambda: defaultdict(int))         

    for k1 in h.keys():
        sum = 0
        for k2 in h[k1].keys():
            sum += h[k1][k2]
        if sum >= 10:
            x[k1] = h[k1]
    return x

def serializedPreprocessedData():
    fp = open("C:\\Users\\novan\\OneDrive\\Desktop\\CODE\\twitter-geotagging\\data\\preprocessed3.csv", 'r')
    a = open("C:\\Users\\novan\\OneDrive\\Desktop\\CODE\\twitter-geotagging\\data\\further2.csv", 'w')
    freq = defaultdict(lambda: defaultdict(int))
    for i in fp:
        x = i[:-1]
        x = x.split('\t')
        keys = x[2].split(',')
        for j in keys:
            freq[j][x[0]] += 1

    freq = deleteUnnecessaryWord(freq)

    attr = sorted(freq.keys())
    a.write("{},{},{}\n".format("ID", ','.join(attr), "Class"))
    fp.seek(0, 0)
    for i in fp:
        x = i[:-1].split('\t')
        a.write("{}".format(x[0]))
        for y in attr:
            a.write(",{}".format(freq[y][x[0]]))
        a.write(",{}\n".format(x[1]))


    fp.close()
    a.close()

# %%
filepath = "C:\\Users\\novan\\OneDrive\\Desktop\\CODE\\twitter-geotagging\\data\\train-raw.tsv"
preprocess(filepath)
serializedPreprocessedData()

# %%
from copy import copy
def fitTestData(fp: str, preprocessed: str):
    data = open(fp, 'r')
    preprocessed = open(preprocessed, 'r')
    header = preprocessed.readline()
    header = header.split(',')[1:-1]
    ids = []
    vals = {}
    y = []
    for i in header:
        vals[i] = []
    data.readline()
    for row in data:
        x = row.split('\t')
        keys = extractKeywords(x[2])
        a = {}
        ids.append(x[0])
        for i in keys:
            if i not in a:
                a[i] = 1
            else:
                a[i] += 1
        
        for i in vals.keys():
            if i in a:
                vals[i].append(a[i])
            else:
                vals[i].append(0)
        y.append(x[1])
    df = pd.DataFrame(vals)

    return ids, df, y

def fitTrainData(data: str):
    data = pd.read_csv(data)
    return data.iloc[:, 1:-1], data.iloc[:, -1]  



# %%
from sklearn.naive_bayes import MultinomialNB
# from sklearn.feature_extraction import DictVectorizer
preprocessed = "C:\\Users\\novan\\OneDrive\\Desktop\\CODE\\twitter-geotagging\\data\\further2.csv"
_, X_test, y_test = fitTestData("C:\\Users\\novan\\OneDrive\\Desktop\\CODE\\twitter-geotagging\\data\\dev-raw.tsv", preprocessed)
X_train, y_train = fitTrainData(preprocessed)
classifier = MultinomialNB()
head1 = sorted(list(X_train.columns.values))
head2 = sorted(list(X_test.columns.values))
if head1 != head2:
    print("What the fuck")

classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)
print("Score: {}\n".format(score))

# %%
ids, X_actualTest, _ = fitTestData("C:\\Users\\novan\\OneDrive\\Desktop\\CODE\\twitter-geotagging\\data\\test-raw.tsv", preprocessed)
predict = classifier.predict(X_actualTest)
pd.DataFrame({"Id": ids, "Class": list(predict)}).to_csv("C:\\Users\\novan\\OneDrive\\Desktop\\CODE\\twitter-geotagging\\data\\result.csv", index=False)

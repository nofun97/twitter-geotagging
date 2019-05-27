# %%
import pandas as pd
import re
from textblob import TextBlob, Word
from nltk.tokenize import TweetTokenizer
from collections import defaultdict
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

def serializedPreprocessedData(tokenizedCSV: str, furtherPreprocess: str):
    fp = open(tokenizedCSV, 'r')
    a = open(furtherPreprocess, 'w')
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
tokenizedCSV = "C:\\Users\\novan\\OneDrive\\Desktop\\CODE\\twitter-geotagging\\data\\preprocessed4.csv"
furtherPreprocess = "C:\\Users\\novan\\OneDrive\\Desktop\\CODE\\twitter-geotagging\\data\\further2.csv"
    
# %%
preprocess(filepath, tokenizedCSV)
serializedPreprocessedData(tokenizedCSV, furtherPreprocess)

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
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier
# from sklearn.feature_extraction import DictVectorizer
_, X_test, y_test = fitTestData("C:\\Users\\novan\\OneDrive\\Desktop\\CODE\\twitter-geotagging\\data\\dev-raw.tsv", furtherPreprocess)
X_train, y_train = fitTrainData(furtherPreprocess)
classifiers = {"MultinomialNB": MultinomialNB(), "DecisionTree": DecisionTreeClassifier(), "BernoulliNB": BernoulliNB(), "GaussianNB": GaussianNB()}
for i in classifiers:
    classifiers[i].fit(X_train, y_train)
    score = classifiers[i].score(X_test, y_test)
    print("Classifier: {}, Score: {}\n".format(i, score))

# %% [markdown]
# ## Results
# 1. Classifier: MultinomialNB, Score: 0.30338192732340014
# 2. Classifier: DecisionTree, Score: 0.30348911994854755, Kaggle Score: 0.28988
# 3. Classifier: BernoulliNB, Score: 0.30710687104727197, Kaggle Score: 0.29247
# 4. Classifier: GaussianNB, Score: 0.14060992603708866


# %%
ids, X_actualTest, _ = fitTestData("C:\\Users\\novan\\OneDrive\\Desktop\\CODE\\twitter-geotagging\\data\\test-raw.tsv", furtherPreprocess)
predict = classifiers["DecisionTree"].predict(X_actualTest)
pd.DataFrame({"Id": ids, "Class": list(predict)}).to_csv("C:\\Users\\novan\\OneDrive\\Desktop\\CODE\\twitter-geotagging\\data\\result5.csv", index=False)

# %%
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# %%
_, X_test, y_test = fitTestData("C:\\Users\\novan\\OneDrive\\Desktop\\CODE\\twitter-geotagging\\data\\dev-raw.tsv", furtherPreprocess)
X_train, y_train = fitTrainData(furtherPreprocess)

X_train = X_train.astype('int')
X_test = X_test.astype('int')

encoder = LabelEncoder()
encoder.fit(y_train)
encodedTrain = encoder.transform(y_train)
encodedTest = encoder.transform(y_test)

kerasTrain = tf.keras.utils.to_categorical(encodedTrain)
kerasTest = tf.keras.utils.to_categorical(encodedTest)

nCols = X_train.shape[1]

#%%
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(200, activation='relu', input_shape=(nCols,)))
model.add(tf.keras.layers.Dense(200, activation='relu'))
model.add(tf.keras.layers.Dense(200, activation='relu'))
model.add(tf.keras.layers.Dense(1))

model.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])

early_stopping_monitor = tf.keras.callbacks.EarlyStopping(patience=3)

model.fit(X_train, kerasTrain, epochs=30, callbacks=[early_stopping_monitor])
test_loss, test_acc = model.evaluate(X_test, kerasTest)
print("Tensorflow Keras, Accuracy: {}\n".format(test_acc))
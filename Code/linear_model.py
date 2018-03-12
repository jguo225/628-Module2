from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
import pandas as pd
import re
import numpy as np
from sklearn.preprocessing import scale
trainname = 'first2000.csv'
wnl = WordNetLemmatizer()
stemmer = SnowballStemmer("english")
# stemmer = PorterStemmer()
stoplist = set(stopwords.words("english"))

def process_review(review):
    new = []
    for rev in review:
        a = re.sub(r'[^a-zA-Z]', ' ', rev)
        a = a.lower()
        # texts = [stemmer.stem(word) for word in review.lower().split() if word not in stoplist]
        texts = [wnl.lemmatize(word) for word in a.split() if word not in stoplist]
        # texts = [word for word in review.lower().split() if word not in stoplist]
        new.append(texts)
    return new
def cat_process(cats):
    newcats = []
    for cat in cats:
        a = re.sub(r'[^a-zA-Z,]', '', cat)
        # a = re.sub(r'[\'\[\]]', ' ', cat)
        b = re.sub(r'[,]', ' ', a).split()
        newcats.append(b)
    return newcats



word_dic = {}
cat_dic = {}

train = pd.read_csv(trainname)

z1 = train.loc[:, 'categories']
tz1 = cat_process(z1)
x1 = train.loc[:, 'text']
tx1 = process_review(x1)
a1 = train.loc[:, 'longitude']
b1 = train.loc[:, 'latitude']

y1 = train.loc[:, 'stars']

columns = []

# Convert the submission dates column to datetime.
reviews_dates = pd.to_datetime(train["date"])

# Transform functions for the datetime column.
transform_functions = [
    lambda x: x.year,
    lambda x: x.month,
    lambda x: x.day,
]

# Apply all functions to the datetime column.
for func in transform_functions:
    columns.append(reviews_dates.apply(func))

# Convert the meta features to a numpy array.
date = np.asarray(columns).T
year = scale(date[:,0])
month = scale(date[:,0])
date = scale(date[:,0])


for index in range(len(tx1)):
    for word in tx1[index]:
        if word in word_dic:
            word_dic[word][0] += y1[index]
            word_dic[word][1] += 1
        else:
            word_dic[word] = [y1[index], 1]

for i in range(len(tz1)):
    for cat in tz1[i]:
        if cat in cat_dic:
            cat_dic[cat][0] += y1[i]
            cat_dic[cat][1] += 1
        else:
            cat_dic[cat] = [y1[i], 1]



score1 = []
y3 = []
for index in range(len(tx1)):
    s = 0
    n = 0
    for word in tx1[index]:
        if word in word_dic:
            s += float(word_dic[word][0]) / word_dic[word][1]
            n += 1
    if n == 0:
        continue
    y3.append(y1[index])
    score1.append(s/n)
y4 = []
score2 = []
for index in range(len(tz1)):
    s = 0
    n = 0
    for cat in tz1[index]:
        if cat in cat_dic:
            s += float(cat_dic[cat][0]) / cat_dic[cat][1]
            n += 1
    if n == 0:
        continue
    y4.append(y3[index])
    score2.append(s/n)

a1 = scale(a1.values)
b1 = scale(b1.values)
score1 = scale(score1)
score2 = scale(score2)
X = np.column_stack((score1, score2))
X = np.column_stack((X, a1, b1, year, month, date))
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X, y4)

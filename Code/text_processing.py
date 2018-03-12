from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import re
import numpy as np
trainname = 'train_data.csv'
testname = 'testval_data.csv'
wnl = WordNetLemmatizer()
stemmer = SnowballStemmer("english")
# stemmer = PorterStemmer()
stoplist = set(stopwords.words("english"))

# remove all the punctuation, whitespace and stop words, convert all the disparities of a word into their normalized form.
def process_review(review):
    review = re.sub(r'[^a-zA-Z]', ' ', review)
    review = review.lower()
    # texts = [stemmer.stem(word) for word in review.lower().split() if word not in stoplist]
    texts = [wnl.lemmatize(word) for word in review.lower().split() if word not in stoplist]
    # texts = [word for word in review.lower().split() if word not in stoplist]
    return texts

# Our list of functions to apply.
transform_functions = [
    lambda x: len(x),
    lambda x: x.count(" "),
    lambda x: x.count("."),
    lambda x: x.count("!"),
    lambda x: x.count("?"),
    lambda x: len(x) / (x.count(" ") + 1),
    lambda x: x.count(" ") / (x.count(".") + 1),
    lambda x: len(re.findall("\d", x)),
    lambda x: len(re.findall("[A-Z]", x)),
]

# Apply each function and put the results into a list.
columns = []
for func in transform_functions:
    columns.append(reviews["text"].apply(func))

# Convert the meta features to a numpy array.
meta = np.asarray(columns).T


# TfidfVectorizer
tfv = TfidfVectorizer(analyzer='word',min_df=3,ngram_range=(1, 2), smooth_idf=1,stop_words=None, strip_accents=None, sublinear_tf=1,token_pattern=r'\w{1,}', use_idf=1).fit(x1)


# CountVectorizer
train = pd.read_csv(trainname)
test = pd.read_csv(testname)
x1 = train.loc[:, 'text']
x2 = test.loc[:, 'text']
cvt = CountVectorizer(analyzer=process_review).fit(x1)
tx1 = cvt.transform(x1)
tx2 = cvt.transform(x2)
# np.savetxt("x.txt", tx.toarray(), delimiter=",")
y1 = train.loc[:, 'stars']

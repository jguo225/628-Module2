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
# y2 = test.loc[:, 'stars']
#############################################
# Logistic
from sklearn.linear_model import LogisticRegressionCV
lg = LogisticRegressionCV(class_weight='balanced')
lg.fit(tx1, y1)
preds2 = lg.predict(tx2)
np.savetxt("logistic5.csv",preds2,fmt = '%d', delimiter=",")

# SVM
from sklearn import svm
from sklearn.grid_search import GridSearchCV
clf = svm.SVR()
clf.fit(tx1,y1)
preds1 = clf.predict(tx1)
preds2 = clf.predict(tx2)

# Grid Search
def svm_cross_validation(train_x, train_y):
    model = SVC(kernel='rbf', probability=True)
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'gamma': [0.1, 0.001, 0.0001,0.00001]}
    grid_search = GridSearchCV(model, param_grid, verbose=1)
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in list(best_parameters.items()):
        print(para, val)
    model = SVR(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
    model.fit(train_x, train_y)
    return model

# SVD
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components = 700)
svd.fit(tx1)
tx1 = svd.transform(tx1)
tx2 = svd.transform(tx2)

# GBM
from sklearn.ensemble import GradientBoostingRegressor
gbm = GradientBoostingRegressor()
gbm.fit(tx1, y1)
preds1 = lg.predict(tx1)

# knn
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors=50)
knn.fit(tx1,y1)
# preds1 = knn.predict(tx1)
preds2 = knn.predict(tx2)

# naive_bayes
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(tx1, y1)
preds1 = nb.predict(tx1)

# neural network
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(tx1)
tx1 = scaler.transform(tx1)
tx2 = scaler.transform(tx2)
from sklearn.neural_network import MLPRegressor
nn = MLPRegressor(hidden_layer_sizes=(50,5,5), alpha = 10, activation = 'tanh', learning_rate = 'invscaling')
nn.fit(tx1,y1)

# grid search
from sklearn.preprocessing import label_binarize
y1 = label_binarize(y1, classes=[1, 2, 3, 4, 5])
param_test1 = {'n_estimators':[1050，1070]}
gsearch1 = GridSearchCV(estimator = RandomForestRegressor(min_samples_split=100,
                                  min_samples_leaf=20,max_depth=8, max_features = 'log2' ,random_state=10),
                       param_grid = param_test1, scoring='roc_auc',cv=5)
gsearch1.fit(tx1,y1)

# random forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
rf0 = RandomForestRegressor(n_estimators= 400, max_depth=25,
                                  min_samples_leaf=5, max_features='log2',oob_score=True, random_state=10)
rf0.fit(tx1,y1)
preds1 = rf0.predict(tx1)
preds2 = rf0.predict(tx2)
from sklearn.metrics import mean_squared_error
from math import sqrt
c = sqrt(mean_squared_error(preds1, y1))
d = sqrt(mean_squared_error(preds2, y2))

# ridge
from sklearn.linear_model import RidgeCV
clf = RidgeCV(alphas=(0.01,0.1,1.0,10.0,100.0))
clf.fit(tx1,y1)

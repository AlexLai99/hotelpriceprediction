import pandas as pd
data = pd.read_csv('listings.csv')
print data
pd.set_option('display.max_rows', 5)
pd.set_option('display.max_columns', 97)
print data
cols = ['price','accommodates','bedrooms','beds','latitude','longitude','neighbourhood_cleansed','room_type','cancellation_policy','instant_bookable','reviews_per_month','number_of_reviews','availability_30','review_scores_rating']
data = pd.read_csv('listings.csv', usecols=cols)

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter	
from matplotlib.pyplot import figure
import seaborn as sns

sns.set(font_scale = 1)
sns.jointplot(x=data. longitude.values, y=data. latitude.values, size=8)
plt.ylabel('Latitude')
plt.xlabel('Longitude')
sns.despine
plt.savefig("Hotel Location Distribution")
plt.figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
data['neighbourhood_cleansed'].value_counts().plot(kind='bar')
plt.title("Neighbourhood_Cleansed ",fontsize=16)
plt.xlabel('Neighbourhood_Cleansed ', fontsize=10)
plt.ylabel('Counts', fontsize=16)
plt.savefig("Neighbourhood_Cleansed")

plt.figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
data['bedrooms'].value_counts().plot(kind='bar')
plt.title('Number of Bedrooms',fontsize=16)
plt.xlabel('Bedrooms',fontsize=16)
plt.ylabel('Count',fontsize=16)
sns.despine
plt.savefig("Number of Bedrooms")
data['reviews_per_month'].fillna(0, inplace=True)

data = data[data.bedrooms != 0]
data = data[data.beds != 0]
data = data[data.price != 0]
data = data.dropna(axis=0)
data = data[data.bedrooms == 1]

data['price'] = data['price'].replace('[\$,)]','',  \
        regex=True).replace('[(]','-', regex=True).astype(float)
nc_dummies = pd.get_dummies(data.neighbourhood_cleansed)
rt_dummies = pd.get_dummies(data.room_type)
cp_dummies = pd.get_dummies(data.cancellation_policy)

ib_dummies = pd.get_dummies(data.instant_bookable, prefix="instant")
ib_dummies = ib_dummies.drop('instant_f', axis=1)

alldata = pd.concat((data.drop(['neighbourhood_cleansed', \
    'room_type', 'cancellation_policy', 'instant_bookable'], axis=1), \
    nc_dummies.astype(int), rt_dummies.astype(int), \
    cp_dummies.astype(int), ib_dummies.astype(int)), \
    axis=1)
allcols = alldata.columns

scattercols = ['price','accommodates', 'number_of_reviews', 'reviews_per_month', 'beds', 'latitude','longitude','availability_30', 'review_scores_rating']
pd.scatter_matrix(data[scattercols], figsize=(12, 12), c='red', alpha=0.2)
plt.savefig('Scatter Matrix')

mport sklearn.metrics as metrics
from sklearn import linear_model
from sklearn.cross_validation import train_test_split

rs = 1
ests = [ linear_model.LinearRegression(), linear_model.Ridge(), linear_model.Lasso(), linear_model.ElasticNet(), linear_model.BayesianRidge(), linear_model.OrthogonalMatchingPursuit() ]
ests_labels = np.array(['Linear', 'Ridge', 'Lasso', 'ElasticNet', 'BayesRidge', 'OMP'])
errvals = np.array([])

x_train, x_test, y_train, y_test = train_test_split(alldata.drop(['price'], axis=1), alldata.price, test_size=0.2, random_state=20)
for e in ests:
    e.fit(x_train, y_train)
    this_err = metrics.median_absolute_error(y_test, e.predict(x_test))
errvals = np.append(errvals, this_err)

pos = np.arange(errvals.shape[0])
srt = np.argsort(errvals)
plt.figure(figsize=(7,5))
plt.bar(pos, errvals[srt], align='center')
plt.xticks(pos, ests_labels[srt])
plt.xlabel('Estimator')
plt.ylabel('Median Absolute Error')
plt.savefig('Estimator')

from sklearn.grid_search import GridSearchCV
from sklearn import ensemble
from sklearn import preprocessing
n_est = 600
tuned_parameters = {"n_estimators": [ n_est ],"max_depth" : [ 4 ],"learning_rate": [ 0.01 ],"min_samples_split" : [ 1.0 ],"loss" : [ 'ls', 'lad' ]}
gbr = ensemble.GradientBoostingRegressor()
clf = GridSearchCV(gbr, cv=3, param_grid=tuned_parameters,scoring='median_absolute_error')
preds = clf.fit(x_train, y_train)
best = clf.best_estimator_
mar = metrics.median_absolute_error(y_test, clf.predict(x_test))

test_score = np.zeros(n_est, dtype=np.float64)
train_score = best.train_score_
for i, y_pred in enumerate(best.staged_predict(x_test)):
    test_score[i] = best.loss_(y_test, y_pred)
plt.figure(figsize=(12, 12))
plt.subplot(1, 1, 1)
plt.plot(np.arange(n_est), train_score, 'darkblue', label='Training Set Error')
plt.plot(np.arange(n_est), test_score, 'red', label='Test Set Error')
plt.legend(loc='upper right',fontsize=16)
plt.xlabel('Boosting Iterations',fontsize=16)
plt.ylabel('Least Absolute Deviation',fontsize=16)
plt.savefig('Boosting Iterations')

print mar

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

feature_importance = best.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
iv = feature_importance[sorted_idx]
feature = x_train.columns[sorted_idx]
plt.figure(figsize=(16,16))
plt.barh(pos, iv,align='center')
plt.yticks(pos, feature,fontsize=16)
plt.xlabel('Feature Importance',fontsize=20)
plt.title('Variable Importance',fontsize=20)
plt.savefig('Variable Importance')

pd.set_option('display.max_rows', 11)
pd.set_option('display.max_columns', 11)
corrPearson = data.corr()
print corrPearson

ax = plt.figure().add_subplot(figsize=(16,16))
correction=abs(corrPearson)
ax = sns.heatmap(correction,cmap=plt.cm.Reds, linewidths=0.05,vmax=1, vmin=0 ,annot=True,annot_kws={'size':6,'weight':'bold'})
ax.set_title('Correlation Matrix')
ticks = np.arange(0,10,1)
plt.xticks(np.arange(10)+0.5, ['latitude','longitude','accommodates','bedrooms','beds', 'price','availability_30','number_of_reviews','review_scores_rating','reviews_per_month'])
plt.yticks(np.arange(10)+0.5, ['latitude','longitude','accommodates','bedrooms','beds', 'price','availability_30','number_of_reviews','review_scores_rating','reviews_per_month']) ax.set_xticklabels(['latitude','longitude','accommodates','bedrooms','beds', 'price','availability_30','number_of_reviews','review_scores_rating','reviews_per_month'])
ax.set_yticklabels(['latitude','longitude','accommodates','bedrooms','beds', 'price','availability_30','number_of_reviews','review_scores_rating','reviews_per_month'])
plt.savefig('Pearson Correlation Matrix')



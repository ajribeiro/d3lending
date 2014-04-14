# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Mathieu Blondel <mathieu@mblondel.org>
#         Lars Buitinck <L.J.Buitinck@uva.nl>
# License: BSD 3 clause

from __future__ import print_function

import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
import pylab as pl
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier, Perceptron, PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, auc
import random

#read in the historical csv file
df = pd.io.parsers.read_csv('LoanStats3a_securev1.csv',skiprows=1)
df = df.sort(column='list_d')

#iterate through time running the model for each loan
alldata,alltgt = [],[]
train = 1
#store data for plotting
# last_date = '%04d-%02d-%02d' % (start.year,start.month,start.day)

X_train,y_train,X_test,y_test = [],[],[],[]

#collect all of the data which we want
for index,row in df.iterrows():
    try:
        # if row['list_d']
        #check if this data was used for training th model
        if row['home_ownership'] == 'NONE':
            continue

        # if row['grade'] != 'A': continue

        if isinstance(row['desc'],str):
            ll = len(row['desc'])
        else:
            ll = 1

        #data which is used in the model
        tempdata = {
                    'loan_amnt':row['loan_amnt'],
                    'term':float(row['term'].replace('months','')),
                    'int_rate':float(row['int_rate'][:-1]),
                    'installment':row['installment'],
                    'grade':row['grade'],
                    'emp_length':row['emp_length'],
                    'home_ownership':row['home_ownership'],
                    'annual_income':row['annual_inc'],
                    'fico_range_high':row['fico_range_high'],
                    'fico_range_low':row['fico_range_low'],
                    'inq_last_6mths':row['inq_last_6mths'],
                    'pub_rec_bankruptcies':row['pub_rec_bankruptcies'],
                    'len_desc':ll
                    # 'addr_state':row['addr_state']
                    }

        elen = float(row['emp_length'].replace('<','')[:2])
        tt = [
                row['loan_amnt'],
                float(row['term'].replace('months','')),
                float(row['int_rate'][:-1]),
                row['installment'], 
                row['fico_range_high'],
                row['fico_range_low'],
                row['inq_last_6mths'],
                # 'grade':row['grade'],
                # ho,
                row['annual_inc'],
                row['pub_rec_bankruptcies']
                # 'addr_state':row['addr_state']
            ]

        if not np.isfinite(tt).all():
            continue

        if row['list_d'] < '2009-07-01':
            if 'Fully Paid' in row['loan_status']:
                if random.random() < .2:
                    y_train.append(1)
                    X_train.append(tempdata)
            elif 'Charged Off' in row['loan_status']:
                y_train.append(0)
                X_train.append(tempdata)
        else:
            if 'Fully Paid' in row['loan_status']:
                y_test.append(1)
                X_test.append(tempdata)
            elif 'Charged Off' in row['loan_status']:
                y_test.append(0)
                X_test.append(tempdata)
    except:
        pass

vec = DictVectorizer()
X_train = vec.fit_transform(X_train).toarray()
X_test = vec.fit_transform(X_test).toarray()
# X_train, X_test, y_train, y_test = \
#     train_test_split(alldata, alltgt, test_size=0.33, random_state=0)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

###############################################################################
# Benchmark classifiers
def benchmark(clf):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    c = 0
    for i in pred:
        if i == 1:
            c += 1
    print(c,len(pred))
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.f1_score(y_test, pred)
    print("f1-score:   %0.3f" % score)

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

        # if opts.print_top10 and feature_names is not None:
        #     print("top 10 keywords per class:")
        #     for i, category in enumerate(categories):
        #         top10 = np.argsort(clf.coef_[i])[-10:]
        #         print(trim("%s: %s"
        #               % (category, " ".join(feature_names[top10]))))
        print()

    # if opts.print_report:
    #     print("classification report:")
    #     print(metrics.classification_report(y_test, pred,
    #                                         target_names=categories))

    # if opts.print_cm:
    #     print("confusion matrix:")
    #     print(metrics.confusion_matrix(y_test, pred))

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time


results = []
for clf, name in (
        (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
        (Perceptron(n_iter=50), "Perceptron"),
        (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
        (KNeighborsClassifier(n_neighbors=10), "kNN")):
    print('=' * 80)
    print(name)
    results.append(benchmark(clf))

for penalty in ["l2", "l1"]:
    print('=' * 80)
    print("%s penalty" % penalty.upper())
    # Train Liblinear model
    results.append(benchmark(LinearSVC(loss='l2', penalty=penalty,
                                            dual=False, tol=1e-3)))

    # Train SGD model
    results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                           penalty=penalty)))

# Train SGD with Elastic Net penalty
print('=' * 80)
print("Elastic-Net penalty")
results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                       penalty="elasticnet")))

# Train NearestCentroid without threshold
print('=' * 80)
print("NearestCentroid (aka Rocchio classifier)")
results.append(benchmark(NearestCentroid()))

# Train sparse Naive Bayes classifiers
print('=' * 80)
print("Naive Bayes")
results.append(benchmark(MultinomialNB(alpha=.01)))
results.append(benchmark(BernoulliNB(alpha=.01)))


class L1LinearSVC(LinearSVC):

    def fit(self, X, y):
        # The smaller C, the stronger the regularization.
        # The more regularization, the more sparsity.
        self.transformer_ = LinearSVC(penalty="l1",
                                      dual=False, tol=1e-3)
        X = self.transformer_.fit_transform(X, y)
        return LinearSVC.fit(self, X, y)

    def predict(self, X):
        X = self.transformer_.transform(X)
        return LinearSVC.predict(self, X)

print('=' * 80)
print("LinearSVC with L1-based feature selection")
results.append(benchmark(L1LinearSVC()))


# make some plots

indices = np.arange(len(results))

results = [[x[i] for x in results] for i in range(4)]

clf_names, score, training_time, test_time = results
training_time = np.array(training_time) / np.max(training_time)
test_time = np.array(test_time) / np.max(test_time)

pl.figure(figsize=(12,8))
pl.title("Score")
pl.barh(indices, score, .2, label="score", color='r')
pl.barh(indices + .3, training_time, .2, label="training time", color='g')
pl.barh(indices + .6, test_time, .2, label="test time", color='b')
pl.yticks(())
pl.legend(loc='best')
pl.subplots_adjust(left=.25)
pl.subplots_adjust(top=.95)
pl.subplots_adjust(bottom=.05)

for i, c in zip(indices, clf_names):
    pl.text(-.3, i, c)

pl.show()
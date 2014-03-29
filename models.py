import pandas as pd
import datetime as dt
import random
import numpy as np
import pylab as pl
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from  sklearn.linear_model import RidgeClassifierCV
from sklearn import tree,decomposition
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
import cPickle as pickle

df = pd.io.parsers.read_csv('LoanStats3a_securev1.csv',skiprows=1)

train_data,test_data = [],[]
y_train,y_test = [],[]

datalog = open('datalog.txt','w')

for index,row in df.iterrows():

    try: 
        if row['home_ownership'] == 'NONE': continue
        
        elen = float(row['emp_length'].replace('<','')[:2])

        lid = row['id']
        tempdata = {
                    'loan_amnt':row['loan_amnt'],
                    'term':float(row['term'].replace('months','')),
                    'int_rate':float(row['int_rate'][:-1]),
                    'installment':row['installment'],
                    'grade':row['grade'],
                    'emp_length':row['emp_length'],
                    'home_ownership':row['home_ownership'],
                    'annual_income':row['annual_inc'],
                    # 'addr_state':row['addr_state']
                    }

        tt = [
                row['loan_amnt'],
                float(row['term'].replace('months','')),
                float(row['int_rate'][:-1]),
                row['installment'], 

                # 'grade':row['grade'],
                elen,
                # ho,
                row['annual_inc'],
                # 'addr_state':row['addr_state']
                ]

        if not np.isfinite(tt).all():
            continue

        if 'Fully Paid' in row['loan_status']:
            tar = 1
        else:
            tar = 0

    except: 
        continue

    if random.random() < .5:
        train_data.append(tempdata)
        y_train.append(tar)
        datalog.write('%s,0\n' % lid)
    else:
        test_data.append(tempdata)
        y_test.append(tar)
        datalog.write('%s,1\n' % lid)

datalog.close()
print 'done constructing dataset'


# # ########################################
# # ## DO A SINGLE DECISION TREE
# # ########################################

vec = DictVectorizer()

# pickle.dump(X_test, open("X_test.p", "wb"))
# pickle.dump(y_test, open("y_test.p", "wb"))
# X = vec.fit_transform(all_data).toarray()

X_train = vec.fit_transform(train_data).toarray()
X_test = vec.fit_transform(test_data).toarray()

pickle.dump(vec, open("vec.p", "wb"))

# pickle.dump(X, open('x_data.p', 'wb'))
# pickle.dump(all_target, open('all_target.p', 'wb'))

# X = pickle.load(open( "x_data.p", "rb" ))
# vec = pickle.load(open( "vec.p", "rb" ))
# all_target = pickle.load(open( "all_target.p", "rb" ))

# X_train, X_test, y_train, y_test = \
#     cross_validation.train_test_split(X, all_target, \
#     test_size=0.3, random_state=0)


print 'loaded data'

# for clf, name in (
#         (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
#         (Perceptron(n_iter=50), "Perceptron"),
#         (PassiveAggressiveClassifier(n_iter=70), "Passive-Aggressive"),
#         (KNeighborsClassifier(n_neighbors=10), "kNN"),
#         (MultinomialNB(alpha=.01),'multinb'),
#         (LinearSVC(loss='l2', dual=False, tol=1e-3),'linsvc'),
#         (SGDClassifier(alpha=.01, n_iter=50),'sgdc')
#         ):

#     clf.fit(X_train,y_train)

#     y = clf.predict(X_test)

#     n_right,n_wrong = 0,0

#     for i in range(len(y)):
#         if y[i] == 0: continue
#         if y[i] == y_test[i]:
#             n_right += 1
#         else:
#             n_wrong += 1

#     print name,n_right,n_wrong,n_right/float(n_right+n_wrong)

# clf = tree.DecisionTreeClassifier().fit(X_train, y_train)

# y = clf.predict(X_train)

# n_right,n_wrong = 0,0
# for i in range(len(y)):
#     if y[i] == 0: continue
#     if y[i] == y_train[i]:
#         n_right += 1
#     else:
#         n_wrong += 1

# print 'decision tree, train data'
# print n_right,n_wrong,n_right/float(n_right+n_wrong)


# y = clf.predict(X_test)

# n_right,n_wrong = 0,0
# for i in range(len(y)):
#     if y[i] == 0: continue
#     if y[i] == y_test[i]:
#         n_right += 1
#     else:
#         n_wrong += 1
# print 'decision tree, test data'
# print n_right,n_wrong,n_right/float(n_right+n_wrong)


##############################################
## RANDOM FOREST CLASSIFIER
##############################################

from sklearn.cross_validation import cross_val_score

clf = RandomForestClassifier(n_estimators=100)

clf.fit(X_train,y_train)

y = clf.predict(X_train)

n_right,n_wrong = 0,0
for i in range(len(y)):
    if y[i] == 0: continue
    if y[i] == y_train[i]:
        n_right += 1
    else:
        n_wrong += 1

print 'random forest, train data'
print n_right,n_wrong,n_right/float(n_right+n_wrong)

y=clf.predict(X_test)

n_right,n_wrong = 0,0
for i in range(len(y)):
    if y[i] == 0: continue
    if y[i] == y_test[i]:
        n_right += 1
    else:
        n_wrong += 1

print 'random forest, test data'
print n_right,n_wrong,n_right/float(n_right+n_wrong)

pickle.dump(clf, open("clf.p", "wb"))

##############################################
## GB CLASSIFIER
##############################################

# clf = GradientBoostingClassifier(n_estimators=100)

# clf.fit(X_train,y_train)

# y=clf.predict(X_train)

# n_right,n_wrong = 0,0
# for i in range(len(y)):
#     print y[i]
#     if y[i] == 0: continue
#     if y[i] == y_train[i]:
#         n_right += 1
#     else:
#         n_wrong += 1

# print 'gb, train data'
# print n_right,n_wrong,n_right/float(n_right+n_wrong)

# y=clf.predict(X_test)

# n_right,n_wrong = 0,0
# for i in range(len(y)):
#     print y[i]
#     if y[i] == 0: continue
#     if y[i] == y_test[i]:
#         n_right += 1
#     else:
#         n_wrong += 1

# print 'gb, test data'
# print n_right,n_wrong,n_right/float(n_right+n_wrong)

# pickle.dump(vec, open("vec.p", "wb") )
# pickle.dump(clf, open("clf.p", "wb") )

##############################################
## ET CLASSIFIER
##############################################

# clf = RidgeClassifierCV(class_weight={0:1,1:10})

# clf.fit(X_train,y_train)

# y=clf.predict(X_train)

# n_right,n_wrong = 0,0
# for i in range(len(y)):
#     if y[i] == 0: continue
#     if y[i] == y_train[i]:
#         n_right += 1
#     else:
#         n_wrong += 1

# print 'et, train data'
# print n_right,n_wrong,n_right/float(n_right+n_wrong)

# y=clf.predict(X_test)

# n_right,n_wrong = 0,0
# for i in range(len(y)):
#     if y[i] == 0: continue
#     if y[i] == y_test[i]:
#         n_right += 1
#     else:
#         n_wrong += 1

# print 'et, test data'
# print n_right,n_wrong,n_right/float(n_right+n_wrong)







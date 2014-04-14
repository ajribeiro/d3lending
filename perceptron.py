# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Mathieu Blondel <mathieu@mblondel.org>
#         Lars Buitinck <L.J.Buitinck@uva.nl>
# License: BSD 3 clause


import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import random
from sklearn.linear_model import LogisticRegression
from lc_utils import *

features=['loan_amnt','term','int_rate','installment','grade','emp_length',
            'annual_inc','fico_range_high','fico_range_low','inq_last_6mths',
            'pub_rec_bankruptcies','desc','sub_grade']

X_train,X_test,y_train,y_test,rets,vec = get_lc_data(features=features)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

penalty = 'l2'
clf = LogisticRegression(penalty=penalty)
probas = clf.fit(X_train, y_train)
pred = clf.predict(X_test)
score = metrics.f1_score(y_test, pred)
print "f1-score:   %0.3f" % score 

if hasattr(clf, 'coef_'):
    print "dimensionality: %d" % clf.coef_.shape[1]
    print "density: %f" % density(clf.coef_)

cnt,rl = 0,0
for i in range(len(pred)):
    if pred[i] == 0:
        cnt += 1
    if y_test[i] == 0:
        rl += 1

print rl,cnt


# # Compute ROC curve and area the curve
pred = clf.predict_proba(X_test)

xx,yc,ym = [],[],[]
for thresh in [.5+.05*i for i in range(10)]:

    mcors,mtots,cors,tots,retsa,retsb,retsc = [],[],[],[],[],[],[]
    amts,mamts = [],[]
    for ii in range(10):
        cor,tot,mcor,mtot = 0.,0.,0.,0.
        budg = 10000.
        budga,budgb = budg,budg
        obudga,obudgb = budga,budgb
        reta,retb = 0.,0.
        for i in range(len(y_test)):
            #predicted no default, invest
            inv_amnt = (int((pred[i][1]-.5)/.05)+1.)*25.
            if inv_amnt > budga and budga > 0.:
                inv_amnt = budga
            if pred[i][1] > thresh and budga >= inv_amnt:
                tot += 1
                if y_test[i] == 1:
                    cor += 1
                budga -= inv_amnt
                ratio = inv_amnt/X_test[i][vec.get_feature_names().index('loan_amnt')]
                reta += rets[i]*ratio

            #random investment
            inv_amnt = 50.
            if random.random() < .3 and budgb >= inv_amnt:
                mtot += 1
                if y_test[i] == 1:
                    mcor += 1
                budgb -= inv_amnt
                ratio = inv_amnt/X_test[i][vec.get_feature_names().index('loan_amnt')]
                retb += rets[i]*ratio

        if budga < obudga and budgb < obudgb:
            amts.append((obudga-budga))
            mamts.append((obudgb-budgb))
            cors.append(cor/tot)
            tots.append(tot)
            mtots.append(mtot)
            mcors.append(mcor/mtot)
            retsa.append(reta/(obudga-budga))
            retsb.append(retb/(obudgb-budgb))
            retsc.append(reta/(obudga-budga)-retb/(obudgb-budgb))

    if len(cors) == 0: continue

    print 'thresh:',thresh
    print 'model did not default',np.mean(cors),'amount',np.mean(amts)
    print 'market did not default',np.mean(mcors),'amount',np.mean(mamts)
    print 'model return',np.mean(retsa)
    print 'market return', np.mean(retsb)
    print 'returns vs market',np.mean(retsc)
    print

    xx.append(thresh)
    yc.append(np.mean(retsa))
    ym.append(np.mean(retsb))


# fpr, tpr, thresholds = roc_curve(y_test, pred[:, 1],pos_label=1)
# roc_auc = auc(fpr, tpr)
# print("Area under the ROC curve : %f" % roc_auc)

# # Plot ROC curve
# f = pl.figure()
# pl.clf()
# pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
# pl.plot([0, 1], [0, 1], 'k--')
# pl.xlim([0.0, 1.0])
# pl.ylim([0.0, 1.0])
# pl.xlabel('False Positive Rate')
# pl.ylabel('True Positive Rate')
# pl.title('Receiver operating characteristic example')
# pl.legend(loc="lower right")
# # pl.show()
# # f.show()



f = plt.figure()
plt.plot(xx,yc,'b')
plt.plot(xx,ym,'g')
f.show()
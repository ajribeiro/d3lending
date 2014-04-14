import numpy as np
import pylab as pl
from sklearn import neighbors
import random
from lc_utils import *
import matplotlib.pyplot as plt

features=['loan_amnt','term','int_rate','installment','grade','emp_length',
            'annual_inc','fico_range_high','fico_range_low','inq_last_6mths',
            'pub_rec_bankruptcies','desc','sub_grade']

X_train,X_test,y_train,y_test,rets,vec = get_lc_data(features=features, 
        type='regression')

knn = neighbors.KNeighborsRegressor(10)
knn = knn.fit(X_train, y_train)

pred = knn.predict(X_test)


xx,yr,ym = [],[],[]
tlist = [1.+.01*i for i in range(1,20)]

for thresh in tlist:

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
            inv_amnt = (int((pred[i]-1.)/.01)+1.)*15.
            # inv_amnt = 50.
            if inv_amnt > budga and budga > 0:
                inv_amnt = budga
            if pred[i] > thresh and budga >= inv_amnt:
                tot += 1.
                if y_test[i] == 1:
                    cor += 1.
                budga -= inv_amnt
                ratio = inv_amnt/X_test[i][vec.get_feature_names().index('loan_amnt')]
                reta += rets[i]*ratio

            #random investment
            inv_amnt = 50.
            if random.random() < .9 and budgb >= inv_amnt:
                mtot += 1.
                if y_test[i] == 1:
                    mcor += 1.
                budgb -= inv_amnt
                ratio = inv_amnt/X_test[i][vec.get_feature_names().index('loan_amnt')]
                retb += rets[i]*ratio

        if budga < obudga and budgb < obudgb:
            cors.append(cor/tot)
            tots.append(tot)
            mtots.append(mtot)
            mcors.append(mcor/mtot)
            retsa.append(reta/(obudga-budga))
            retsb.append(retb/(obudgb-budgb))
            retsc.append(reta/(obudga-budga)-retb/(obudgb-budgb))
            amts.append(obudga-budga)
            mamts.append(obudgb-budgb)

    if len(cors) == 0: continue

    print 'thresh:',thresh
    print 'model did not default',np.mean(cors),'amount',np.mean(amts)
    print 'market did not default',np.mean(mcors),'amount',np.mean(mamts)
    print 'model return',np.mean(retsa)
    print 'market return', np.mean(retsb)
    print 'returns vs market',np.mean(retsc)
    print

    xx.append(thresh)
    yr.append(np.mean(retsa))
    ym.append(np.mean(retsb))



f = plt.figure()
plt.plot(xx,yr,'b')
plt.plot(xx,ym,'g')
f.show()
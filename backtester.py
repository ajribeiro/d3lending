import pandas as pd
import datetime as dt
import random
import json
import numpy as np
import cPickle as pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier

def backtester(start,end,models=['Market']):
    dataclass = {'0':[],'1':[]}
    datalog = open('datalog.txt','r')
    for line in datalog:
        cols = line.split(',')
        dataclass[cols[1][0]].append(cols[0])
    datalog.close()
    # return dataclass

    #read in the historical csv file
    df = pd.io.parsers.read_csv('LoanStats3a_securev1.csv',skiprows=1)

    df = df.sort(column='list_d')

    #iterate through time running the model for each loan
    amount_loaned,amount_earned = 0.,0.
    n_funded,n_defaulted = 0,0

    #store data for plotting
    last_date = '%04d-%02d-%02d' % (start.year,start.month,start.day)

    retdata = []

    train_data,train_tar,test_data,test_tar = [],[],[],[]
    eph_data = []

    #collect all of the data which we want
    for index,row in df.iterrows():
        try:
            #check if this data was used for training th model
            if row['home_ownership'] == 'NONE':
                continue

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
                        # 'addr_state':row['addr_state']
                        }

            elen = float(row['emp_length'].replace('<','')[:2])
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

            #data whih is also useful but not used in the model
            exdata = {
                'list_d':row['list_d'],
                'total_pymnt':row['total_pymnt'],
                'last_pymnt_d':row['last_pymnt_d'],
                'issue_d':row['issue_d'],
                'loan_status':row['loan_status']
                }

            if 'Fully Paid' in row['loan_status']:
                tar = 1
            else:
                tar = 0

        except:
            continue

        if row['list_d'] <= '%04d-%02d-%02d' % (end.year,end.month,end.day):
            if row['loan_status'] == 'Current':
                continue
            train_data.append(tempdata)
            train_tar.append(tar)
        else:
            eph_data.append(exdata)
            test_data.append(tempdata)
            test_tar.append(tar)

    #dict to hold data to be returned
    retdata = {}

    # print len(alldata)
    #iterate through the models
    for m in models:

        out_data,y = [],[]

        #check what models 
        if m == 'Safest':
            for i in range(len(test_data)):
                if test_data[i]['grade'] == 'A' or test_data[i]['grade'] == 'B':
                    y.append(1)
                else:
                    y.append(0)

        elif m == 'Market':
            y = [1]*len(test_data)

        elif m == 'Random Forest':

            vec = DictVectorizer()
            print 'start load'

            X_train = vec.fit_transform(train_data).toarray()
            X_test = vec.fit_transform(test_data).toarray()

            clf = RandomForestClassifier(n_estimators=100)
            clf.fit(X_train,train_tar)

            print 'done train'

            y=clf.predict(X_test)

            print 'done predict'

        total_spent,total_received = 1e-16,0.
        cnt = 0
        for i in range(len(test_data)):

            if y[i] == 1:
                if eph_data[i]['loan_status'] == 'Current':
                    issue = eph_data[i]['issue_d']
                    syear = int(issue[0:4])
                    smon = issue[5:7]
                    if smon[0] == '0': smon = smon[1:]
                    smon = int(smon)
                    sday = issue[8:10]
                    if sday[0] == '0': sday = sday[1:]
                    sday = int(sday)

                    issue = eph_data[i]['last_pymnt_d']
                    eyear = int(issue[0:4])
                    emon = issue[5:7]
                    if emon[0] == '0': emon = emon[1:]
                    emon = int(emon)
                    eday = issue[8:10]
                    if eday[0] == '0': eday = eday[1:]
                    eday = int(eday)


                    s_d = dt.datetime(syear,smon,sday)
                    e_d = dt.datetime(eyear,emon,eday)

                    delt = (e_d-s_d).days // 30
                    total_spent += test_data[i]['loan_amnt']*float(delt)/test_data[i]['term']
                    
                else:
                    total_spent += test_data[i]['loan_amnt']

                total_received += eph_data[i]['total_pymnt']

            if eph_data[i]['list_d'] != last_date:
                last_date = eph_data[i]['list_d']
                cnt += 1
                if cnt % 14 == 0:
                    out_data.append({'date':eph_data[i]['list_d'].replace('-','/'),'ret':total_received,'inv':total_spent})


        if cnt % 14 != 0:
            out_data.append({'date':eph_data[i]['list_d'].replace('-','/'),'ret':total_received,'inv':total_spent})

        retdata[m] = out_data

    # print n_funded,n_defaulted
    return json.dumps(retdata)

s = dt.datetime(2008,1,1)
e = dt.datetime(2010,1,1)
x = backtester(s,e,['Random Forest'])
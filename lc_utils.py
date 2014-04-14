from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import Imputer
import pandas as pd
import numpy as np
import random

features=None
type='classification'
split_date='2010-07-01'
split_pct=.4
def get_lc_data(split_date='2010-07-01',type='classification',split_pct=.4,
                features=None):

    if features is None:
        features = ['loan_amnt','term','int_rate','installment','grade','emp_length',
                    'annual_inc','fico_range_high','fico_range_low','inq_last_6mths',
                    'desc']


    #read in the historical csv file
    df = pd.io.parsers.read_csv('LoanStats3a_securev1.csv',skiprows=1)
    df = df.sort(column='list_d')

    #iterate through time running the model for each loan
    alldata,alltgt = [],[]               
    #store data for plotting
    # last_date = '%04d-%02d-%02d' % (start.year,start.month,start.day)

    X_train,y_train,X_test,y_test = [],[],[],[]
    rets = []

    #clean the data a bit
    for index,row in df.iterrows():
        if not isinstance(row['loan_status'],str): continue
        if np.isnan(row['total_pymnt']): continue

        if isinstance(row['desc'],str):
            df['desc'][index] = len(row['desc'])
        else:
            df['desc'][index] = np.nan

        if not isinstance(row['int_rate'],str) and np.isnan(row['int_rate']):
            df['int_rate'][index] = np.nan
        else:
            df['int_rate'][index] = float(row['int_rate'][:-1])

    #collect all of the data which we want
    for index,row in df.iterrows():
        if not isinstance(row['loan_status'],str): continue
        if np.isnan(row['total_pymnt']): continue
        tempdata = {}
        for f in features:
            tempdata[f] = row[f]

        if row['list_d'] < split_date:
            if 'Fully Paid' in row['loan_status']:
                if random.random() < split_pct:
                    X_train.append(tempdata)
                    if type == 'classification':
                        y_train.append(1)
                    elif type == 'regression':
                        y_train.append(row['total_pymnt']/row['loan_amnt'])
            elif 'Charged Off' in row['loan_status']:
                X_train.append(tempdata)
                if type == 'classification':
                    y_train.append(0)
                elif type == 'regression':
                    y_train.append(row['total_pymnt']/row['loan_amnt'])
        else:
            if 'Fully Paid' in row['loan_status']:
                X_test.append(tempdata)
                if type == 'classification':
                    y_test.append(1)
                elif type == 'regression':
                    y_test.append(row['total_pymnt']/row['loan_amnt'])
                rets.append(row['total_pymnt'])
            elif 'Charged Off' in row['loan_status']:
                X_test.append(tempdata)
                if type == 'classification':
                    y_test.append(0)
                elif type == 'regression':
                    y_test.append(row['total_pymnt']/row['loan_amnt'])
                rets.append(row['total_pymnt'])


    vec = DictVectorizer()
    X_train = vec.fit_transform(X_train).toarray()
    X_test = vec.fit_transform(X_test).toarray()

    imp = Imputer(missing_values=np.nan, strategy='mean', axis=0)
    imp.fit(X_train)
    X_train = imp.transform(X_train)
    X_test = imp.transform(X_test)

    return X_train,X_test,y_train,y_test,rets,vec
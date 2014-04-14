import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2., 1.02*height, '%s'%str(height*100)[:5],
                ha='center', va='bottom')



#read in the historical csv file
df = pd.io.parsers.read_csv('LoanStats3a_securev1.csv',skiprows=1)

df = df.sort(column='list_d')
n_paid,n_def = 0., 0.
def_by_grade = {}
lens,lenres = [],[]
for index,row in df.iterrows():

    try:
        if 'Charged Off' in row['loan_status']:
            n_def += 1.
        elif 'Fully Paid' in row['loan_status']:
            # print row['total_rec_prncp'],row['loan_amnt']
            n_paid += 1.
    except:
        pass

    try:
        if isinstance(row['desc'],str):
            l = len(row['desc'])
        else:
            l = 1
        if 'Charged Off' in row['loan_status']:
            res=0
        elif 'Fully Paid' in row['loan_status']:
            # print row['total_rec_prncp'],row['loan_amnt']
            res=1
        lens.append(l)
        lenres.append(res)
    except:
        pass
    if row['grade'] and isinstance(row['grade'],str):
        if row['grade'] not in def_by_grade:
            def_by_grade[row['grade']] = {'tot':0., 'def':0.} 
        try:
            if 'Charged Off' in row['loan_status']:
                def_by_grade[row['grade']]['def'] += 1.
                def_by_grade[row['grade']]['tot'] += 1.
            elif 'Fully Paid' in row['loan_status']:
                def_by_grade[row['grade']]['tot'] += 1.
        except:
            pass


n_tot = n_paid+n_def
print n_paid,n_def,n_def/n_tot

f = plt.figure()
irange = [min(lens),max(lens)]
diff = irange[1]-irange[0]
n_bins = 8
delt = float(diff)/n_bins
defs,tots = [0 for i in range(n_bins)],[0 for i in range(n_bins)]
for i in range(len(lens)):
    bin = int((lens[i]-irange[0]-.01)/delt)
    tots[bin] += 1.
    if lenres[i] == 0:
        defs[bin] += 1.

plt.bar([i for i in range(n_bins)],np.array(defs)/np.array(tots), \
        align='center')
# plt.xticks(np.arange(len(x)),labels)
plt.title('default rate vs length of description')
plt.ylabel('loan default %')
# plt.xlabel(ind)
f.show()
# ################################################
# ## PLOT DEFAULTS VS PAID OFF
# ################################################
# f = plt.figure()
# outcomes = ['paid','default']
# y_pos = range(len(outcomes))
# results = [n_paid/n_tot,n_def/n_tot]

# rects = plt.bar(y_pos,results,align='center')
# plt.xticks(y_pos,outcomes)
# plt.ylabel('%')
# plt.title('loan outcome, all loans')

# autolabel(rects)

# # f.show()

# ################################################
# ## PLOT DEFAULTS BY LOAN GRADE
# ################################################
# f = plt.figure()
# # colors = ['r','g','b','o','c','m','y']
# labels = ['A','B','C','D','E','F','G']
# x,y = [],[]
# for i in range(len(def_by_grade)):
#     print labels.index(def_by_grade.keys()[i])
#     x.append(labels.index(def_by_grade.keys()[i]))
#     y.append(def_by_grade[def_by_grade.keys()[i]]['def']/ \
#         def_by_grade[def_by_grade.keys()[i]]['tot'])

# plt.plot(x,y,'bo')
# plt.xticks(np.arange(len(x)),labels)
# plt.title('default by loan grade')
# plt.ylabel('loan default %')
# plt.xlabel('loan grade')
# f.show()

# ##################################################
# ## DEFAULT RATES FOR OTHER POSSIBLE INDICATORS
# ##################################################
# indicators = ['int_rate','fico_range_high','fico_range_low', 
#                 'loan_amnt', 'annual_inc','inq_last_6mths']

# for ind in indicators:
#     str_inds = set(['int_rate'])
#     if ind in str_inds:
#         irange = [float(df[ind].valid().min()[:-1]), \
#             float(df[ind].valid().max()[:-1])]
#     elif ind == 'annual_inc':
#         irange=[df[ind].valid().min(),100000.]
#     else:
#         irange = [df[ind].valid().min(), df[ind].valid().max()]
#     ibins = 8
#     i_range = irange[1]-irange[0]
#     i_del = i_range/ibins
#     defs,tot = [0.]*ibins, [0.]*ibins

#     for index,row in df.iterrows():

#         if ind in str_inds:
#             if isinstance(row[ind],str):
#                 val = float(row[ind][:-1])
#                 bin = int((val - irange[0]) / i_del)
#                 try:
#                     if 'Charged Off' in row['loan_status']:
#                         defs[bin] += 1
#                         tot[bin] += 1
#                     elif 'Fully Paid' in row['loan_status']:
#                         tot[bin] += 1
#                 except:
#                     pass
#         else:
#             if ind == 'annual_inc' and row[ind] > 1000000.: continue
#             if isinstance(row[ind],float) and not np.isnan(row[ind]):
#                 val = row[ind]
#                 bin = int((val - irange[0]) / i_del)
#                 try:
#                     if 'Charged Off' in row['loan_status']:
#                         defs[bin] += 1
#                         tot[bin] += 1
#                     elif 'Fully Paid' in row['loan_status']:
#                         tot[bin] += 1
#                 except:
#                     pass

#     ################################################
#     ## PLOT DEFAULTS BY INTEREST RATE
#     ################################################
#     f = plt.figure()
#     labels = []
#     for i in range(ibins):
#         labels.append('%.2f' % (irange[0] + i*i_del+i_del/2.))
#     x,y = [],[]
#     for i in range(ibins):
#         x.append(i)
#         if tot[i] > 0:
#             y.append(defs[i]/tot[i])
#         else:
#             y.append(0.)

#     plt.bar(x,y,align='center')
#     plt.xticks(np.arange(len(x)),labels)
#     plt.title('default rate vs %s' % ind)
#     plt.ylabel('loan default %')
#     plt.xlabel(ind)
    # f.show()


# df = df[df['list_d'] > '2009-01-01']

# ############################################################
# ############################################################
# ## PLOT RETURNS 
# ############################################################
# ############################################################
# inv,ret = 0.,0.
# def_by_grade = {}

# for index,row in df.iterrows():

#     try:
#         if 'Charged Off' in row['loan_status'] or 'Fully Paid' in row['loan_status']:
#             inv += row['loan_amnt']
#             ret += row['total_pymnt']
#     except:
#         pass

#     if isinstance(row['grade'],str):
#         if row['grade'] not in def_by_grade:
#             def_by_grade[row['grade']] = {'inv':0., 'ret':0.} 
#         try:
#             if 'Charged Off' in row['loan_status'] or 'Fully Paid' in row['loan_status']:
#                 def_by_grade[row['grade']]['inv'] += row['loan_amnt']
#                 def_by_grade[row['grade']]['ret'] += row['total_pymnt']
#         except:
#             pass



# ################################################
# ## PLOT RETURN % BY LOAN GRADE
# ################################################
# f = plt.figure()
# # colors = ['r','g','b','o','c','m','y']
# labels = ['A','B','C','D','E','F','G']
# x,y = [],[]
# for i in range(len(def_by_grade)):
#     print labels.index(def_by_grade.keys()[i])
#     x.append(labels.index(def_by_grade.keys()[i]))
#     y.append(def_by_grade[def_by_grade.keys()[i]]['ret']/ \
#         def_by_grade[def_by_grade.keys()[i]]['inv'])

# plt.plot(x,y,'bo')
# plt.xticks(np.arange(len(x)),labels)
# plt.title('return by loan grade')
# plt.ylabel('loan return %')
# plt.xlabel('loan grade')
# f.show()

# ##################################################
# ## DEFAULT RATES FOR OTHER POSSIBLE INDICATORS
# ##################################################
# indicators = ['int_rate','fico_range_high','fico_range_low', 
#                 'loan_amnt', 'annual_inc','inq_last_6mths']

# for ind in indicators:
#     str_inds = set(['int_rate'])
#     if ind in str_inds:
#         irange = [float(df[ind].valid().min()[:-1]), \
#             float(df[ind].valid().max()[:-1])]
#     elif ind == 'annual_inc':
#         irange=[df[ind].valid().min(),100000.]
#     else:
#         irange = [df[ind].valid().min(), df[ind].valid().max()]
#     ibins = 8
#     i_range = irange[1]-irange[0]
#     i_del = i_range/ibins
#     inv,ret = [[] for i in range(ibins)],[[] for i in range(ibins)]

#     for index,row in df.iterrows():

#         if ind in str_inds:
#             if isinstance(row[ind],str):
#                 val = float(row[ind][:-1])
#                 bin = int((val - irange[0]) / i_del)
#                 try:
#                     if 'Charged Off' in row['loan_status'] or 'Fully Paid' in row['loan_status']:
#                         inv[bin].append(row['loan_amnt'])
#                         ret[bin].append(row['total_pymnt'])
#                 except:
#                     pass
#         else:
#             if ind == 'annual_inc' and row[ind] > 1000000.: continue
#             if isinstance(row[ind],float) and not np.isnan(row[ind]):
#                 val = row[ind]
#                 bin = int((val - irange[0]) / i_del)
#                 try:
#                     if 'Charged Off' in row['loan_status'] or 'Fully Paid' in row['loan_status']:
#                         inv[bin].append(row['loan_amnt'])
#                         ret[bin].append(row['total_pymnt'])
#                 except:
#                     pass

#     for i in range(ibins):
#         inv[i],ret[i] = np.array(inv[i]),np.array(ret[i])
#     ################################################
#     ## PLOT DEFAULTS BY INTEREST RATE
#     ################################################
#     f = plt.figure()
#     labels = []
#     for i in range(ibins):
#         labels.append('%.2f' % (irange[0] + i*i_del+i_del/2.))
#     x,y,yy = [],[],[]
#     for i in range(ibins):
#         x.append(i)
#         if len(inv[i]) > 0:
#             y.append(np.mean(ret[i]/inv[i]))
#             yy.append(np.std(ret[i]/inv[i]))
#         else:
#             y.append(0.)
#             yy.append(0.)

#     plt.bar(x,y,align='center',yerr=yy,alpha=0.3)
#     plt.xticks(np.arange(len(x)),labels)
#     plt.title('returns vs %s' % ind)
#     plt.ylabel('return %')
#     plt.xlabel(ind)
#     f.show()

import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.feature_selection as fs
import sklearn.preprocessing as pp

#load csv file
data = pd.read_csv('credit_cards.csv')

data.head()

#Task #1.a: Determine if there are missing values
pd.isnull(data).sum()

#Task #1.b: plot the distribution of the class attr
height = data['DEFAULT'].value_counts()
x_axis = np.arange(2)
plt.bar(x_axis, height)
plt.text(0, height[0]+100, str(height[0]))
plt.text(1, height[1]+100, str(height[1]))
plt.title('Default Distribution in Credit Card Clients')
plt.show()

#Task #2.c:  chi-square
chi2 = ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
feat_chi2 = pd.DataFrame(data, columns=chi2)
pays = ['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
for p in pays:
    feat_chi2[p] += 2
chi2_score = fs.chi2(feat_chi2, data['DEFAULT'])

#Task #2.c: rank features based on chi-square score
chi_rank = [['SEX', chi2_score[0][0]], ['EDUCATION', chi2_score[0][1]], ['MARRIAGE', chi2_score[0][2]], ['PAY_1', chi2_score[0][3]], ['PAY_2', chi2_score[0][4]], ['PAY_3', chi2_score[0][5]], ['PAY_4', chi2_score[0][6]], ['PAY_5', chi2_score[0][7]], ['PAY_6', chi2_score[0][8]]]
chi_rank.sort(key=lambda x: x[1], reverse=True)
chi_rank

#Task #2.c: comparison information
com = ['ID', 'LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
feature_com = pd.DataFrame(data, columns=muti)
com_score = fs.mutual_info_classif(feature_com, data['DEFAULT'], random_state=156)

#Task #2.c: rank features based on mutual information score
com_rank = [['ID', com_score[0]], ['LIMIT_BAL', com_score[1]], ['AGE', com_score[2]], ['BILL_AMT1', com_score[3]],['BILL_AMT2', com_score[4]],['BILL_AMT3', com_score[5]],['BILL_AMT4', com_score[6]],['BILL_AMT5', com_score[7]],['BILL_AMT6', com_score[8]], ['PAY_AMT1', com_score[9]], ['PAY_AMT2', com_score[10]], ['PAY_AMT3', com_score[11]], ['PAY_AMT4', com_score[12]], ['PAY_AMT5', com_score[13]], ['PAY_AMT6', com_score[14]]]
com_rank.sort(key=lambda x: x[1], reverse=True)
com_rank

#Task #3.i: plotting highest ranked categorical feature
x = data[chi_rank[0][0]].unique()
x.sort()
no = []
yes = []
fig, ax = plt.subplots()

for item in x:
    n = len(data[ (data['DEFAULT'] == 0) & (data[chi_rank[0][0]] == item)])
    y = len(data[ (data['DEFAULT'] == 1) & (data[chi_rank[0][0]] == item)])
    no.append(n)
    yes.append(y)
    ax.text(item-.5, n+y+100, str(n+y))

ax.bar(x, no, label='no')
ax.bar(x, yes, bottom=no, label='yes')

ax.set_ylabel('People')
ax.set_title('Payment 1 Distribution in Credit Card CLients')

plt.show()

#Task #3ii: plotting lowest ranked categorical feature
x = data[chi_rank[-1][0]].unique()
x.sort()
no = []
yes = []
fig, ax = plt.subplots()

for item in x:
    n = len(data[ (data['DEFAULT'] == 0) & (data[chi_rank[-1][0]] == item)])
    y = len(data[ (data['DEFAULT'] == 1) & (data[chi_rank[-1][0]] == item)])
    no.append(n)
    yes.append(y)
    ax.text(item-.25, n+y+100, str(n+y))

ax.bar(x, no, label='no')
ax.bar(x, yes, bottom=no, label='yes')

ax.set_ylabel('People')
ax.set_title('Marriage Distribution in Credit Card CLients')
ax.legend()

plt.show()

#Task #3.iii: plotting highest ranked numerical feature
x = data[data['DEFAULT'] == 0][muti_rank[0][0]]
y = data[data['DEFAULT'] == 1][muti_rank[0][0]]
plot = plt.hist([x, y], label=['no', 'yes'], bins=10)
plt.legend(loc='upper right')
plt.title(label=muti_rank[0][0] + " Distribution in Credit Cards Clients")
for b in range(10):
    plt.text(plot[1][b], plot[0][0][b] + 100, str(plot[0][0][b]), fontsize='x-small')
    plt.text(plot[1][b]+40000, plot[0][1][b] + 100, str(plot[0][1][b]), fontsize='x-small')
plt.show()

#Task #3.iv: plotting lowest ranked numerical feature
x = data[data['DEFAULT'] == 0][muti_rank[-1][0]]
y = data[data['DEFAULT'] == 1][muti_rank[-1][0]]
plot = plt.hist([x, y], label=['no', 'yes'], bins=10)
plt.legend(loc='upper right')
plt.title(label=muti_rank[-1][0] + " Distribution in Credit Cards Clients")
for b in range(10):
    plt.text(plot[1][b], plot[0][0][b] + 50, str(plot[0][0][b]), fontsize='x-small')
    plt.text(plot[1][b]+1000, plot[0][1][b] + 50, str(plot[0][1][b]), fontsize='x-small')
plt.show()

#Task #4: normalize values of numerical features to the range (0,1)
muti.append('AVG_NET')
numericals = pd.DataFrame(data, columns=muti)
scaler = pp.MinMaxScaler()
norm = scaler.fit_transform(numericals)
j=0
for feat in muti:
    print('Original Range for {0}: ({1}, {2})'.format(feat, numericals[feat].min(), numericals[feat].max()))
    print('Normalized Range for {0}: ({1}, {2})'.format(feat, norm[j].min(), norm[j].max()))
    j += 1
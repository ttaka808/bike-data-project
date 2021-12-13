import math
import numpy as np
from numpy.linalg import inv
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('NYC_Bicycle_Counts_2016_Corrected.csv')

total = list(data["Total"])
queen = list(data["Queensboro Bridge"])
will = list(data["Williamsburg Bridge"])
man = list(data["Manhattan Bridge"])
brook = list(data["Brooklyn Bridge"])
precip = list(data["Precipitation"])
hightemp = list(data["High Temp (°F)"])
lowtemp = list(data["Low Temp (°F)"])

for i in range(len(total)):
    total[i] = int(total[i].replace(',',''))
    queen[i] = int(queen[i].replace(',',''))
    will[i] = int(will[i].replace(',',''))
    man[i] = int(man[i].replace(',',''))
    brook[i] = int(brook[i].replace(',',''))
    tmp = precip[i].replace('T','0')
    tmp = tmp.replace(' (S)','')
    precip[i] = float(tmp)


#problem 1
  #mean/cyclist share
tsum = sum(total)
qsum = sum(queen)
wsum = sum(will)
msum = sum(man)
bsum = sum(brook)

print("Brooklyn %: {}".format(bsum/tsum))
print("Manhattan %: {}".format(msum/tsum))
print("Williamsburg %: {}".format(wsum/tsum))
print("Queensboro %: {}".format(qsum/tsum))

  #mode
daysmax = [0,0,0,0]
names = ["Queen", "Will", "Man", "Brook"]
for i in range(len(total)):
    testmax = [queen[i], will[i], man[i], brook[i]]
    daysmax[testmax.index(max(testmax))] += 1

print(names)
print("Days with most: {}".format(daysmax))

  #median
queenSort = sorted(queen)
willSort = sorted(will)
manSort = sorted(man)
brookSort = sorted(brook)

qmed1 = queenSort[len(total)//2]
qmed2 = queenSort[len(total)//2 + 1]
qmed  = (qmed1 + qmed2) / 2
wmed1 = willSort[len(total)//2]
wmed2 = willSort[len(total)//2 + 1]
wmed = (wmed1 + wmed2) / 2
mmed1 = manSort[len(total)//2]
mmed2 = manSort[len(total)//2 + 1]
mmed = (mmed1 + mmed2) / 2
bmed1 = brookSort[len(total)//2]
bmed2 = brookSort[len(total)//2 + 1]
bmed = (bmed1 + bmed2) / 2

print("Brooklyn median: {}".format(bmed))
print("Manhattan median: {}".format(mmed))
print("Williamsburg median: {}".format(wmed))
print("Queensboro median: {}".format(qmed))

  #standard deviation
brookSD = np.std(brook,ddof=1)
manSD = np.std(man,ddof=1)
willSD = np.std(will,ddof=1)
queenSD = np.std(queen,ddof=1)

print("Brooklyn normalized SD: {}".format(brookSD/(bsum/214)))
print("Manhattan normalized SD: {}".format(manSD/(msum/214)))
print("Williamsburg normalized SD: {}".format(willSD/(wsum/214)))
print("Queensboro normalized SD: {}".format(queenSD/(qsum/214)))

#problem 2
big_X = []
y = []
for index in range(len(precip)):
    big_X.append([hightemp[index],lowtemp[index],precip[index],1.0])
for n in range(len(total)):
    y.append([total[n]])
big_X = np.array(big_X)
y = np.array(y)
beta = np.linalg.inv(big_X.T@big_X) @ big_X.T @ y
y_bar = np.mean(total)
topsum = 0
botsum = 0
for i_tru in range(len(total)):
    y_pred = hightemp[i_tru] * beta[0] + lowtemp[i_tru] * beta[1] + precip[i_tru] * beta[2] + beta[3]
    y_true = total[i_tru]
    topsum += (y_true - y_pred)**2
    botsum += (y_true - y_bar)**2

print("Equation: hightemp * "+str(beta[0])+" + lowtemp * "+str(beta[1])+" +  precip * "+str(beta[2])+" + "+str(beta[3]))
print("r^2: "+str(1-topsum/botsum))

#problem 3
total3 = []
precip3 = []
for i in total:
    total3.append((float)(i))
for j in precip:
    precip3.append((float)(j))
degrees = [1,2,3,4,5]
paramFits = []
d1 = []
d2 = []
d3 = []
d4 = []
d5 = []
for a in degrees:
    matrix = [[val ** i for i in range(a,-1,-1)] for val in total3]
    matrix0 = np.array(matrix)
    matrix1 = np.array(precip3)
    leastsquares = np.linalg.inv((matrix0.T @ matrix0)) @ matrix0.T @ matrix1
    paramFits.append(leastsquares)
plt.scatter(total3,precip3,c = 'red')
total3.sort()
for c in total3:
    if 1 in degrees:
        d1.append(c * paramFits[0][0] + paramFits[0][1])
    if 2 in degrees:
        d2.append((c ** 2) * paramFits[1][0] + c * paramFits[1][1] + paramFits[1][2])
    if 3 in degrees:
        d3.append((c ** 3) * paramFits[2][0] + (c ** 2) * paramFits[2][1] + c * paramFits[2][2] + paramFits[2][3])
    if 4 in degrees:
        d4.append((c ** 4) * paramFits[3][0] + (c ** 3) * paramFits[3][1] + (c ** 2) * paramFits[3][2] + c * paramFits[3][3] + paramFits[3][4])
    if 5 in degrees:
        d5.append((c ** 5) * paramFits[4][0] + (c ** 4) * paramFits[4][1] + (c ** 3) * paramFits[4][2] + (c ** 2) * paramFits[4][3] + c * paramFits[4][4] + paramFits[4][5]) 
plt.xlabel('total cyclists')
plt.ylabel('precip')
plt.title('Polynomial Regression')
plt.plot(total3,d1,c='magenta',label='d=1')
plt.plot(total3,d2,c='green',label='d=2')
plt.plot(total3,d3,c='red',label='d=3')
plt.plot(total3,d4,c='blue',label='d=4')
plt.plot(total3,d5,c='orange',label='d=5')
plt.legend()
plt.show()
print(paramFits)

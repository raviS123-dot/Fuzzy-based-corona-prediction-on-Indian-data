import pandas as pd
import numpy as np
import itertools
import array as arr
import matplotlib.pyplot as plt

df = pd.read_csv("cases_only.csv", sep=r'\s*,\s*', header=0, encoding='ascii', engine='python')
df.drop(['Date'],axis = 1, inplace = True)

data = df.to_numpy()
mean = np.mean(data)

A = []
B = []

for i in range(len(data)):
    if data[i] <= mean:
        A.append(data[i])
    if data[i] >  mean:
        B.append(data[i])

U_A = [min(A),mean]
U_B = [mean,max(B)]

F_A = 0
F_B = 0
i=1

for i in range(len(A)):
    F_A += A[i] - A[abs(i-1)]

i=1
for i in range(len(B)):
    F_B += B[i] - B[abs(i-1)] 

D_F_A = F_A/len(A)
D_F_B = F_B/len(B)

#finding upper and lower limits of data <= mean
L_L_A = []
U_L_A = []

i=0
for i in range(len(A)):
    L_L_A.append(min(A)+ (i)* D_F_A)
    U_L_A.append(min(A)+ (i+1)* D_F_A)

lowerLimit_A = np.array(L_L_A)
upperLimit_A = np.array(U_L_A)

#finding upper and lower limits of data > mean
L_L_B = []
U_L_B = []
i=0

for i in range(len(B)):
    L_L_B.append(mean+ (i)* D_F_B)
    U_L_B.append(mean+ (i+1)* D_F_B)
    
lowerLimit_B = np.array(L_L_B)
upperLimit_B = np.array(U_L_B)

np.savetxt('test1.csv',np.c_[lowerLimit_A,upperLimit_A], fmt="%d",delimiter=',') # to save as csv
np.savetxt('test2.csv',np.c_[lowerLimit_B,upperLimit_B], fmt="%d",delimiter=',') # to save as csv

test1 = np.genfromtxt('test1.csv', delimiter=',')
test2 = np.genfromtxt('test2.csv', delimiter=',')

test1[len(test1)-1,1] = test2[0,0]
test2[len(test2)-1,1] = max(data)

np.savetxt('intervals.csv',np.r_[test1,test2], fmt="%d",delimiter=',') # to save as csv
test3 = np.genfromtxt('intervals.csv',delimiter=',')

test3 = test3.tolist()
data = data.flatten()
data = data.tolist()

# Finding intervals 
interval = []
i=0
j=0
for i in range(len(test3)):
    temp1= []
    for j in range(len(test3)):
        if data[j]>=test3[i][0] and data[j]<=test3[i][1]:
            temp1.append(data[j])
    interval.append(temp1)

interval = [x for x in interval if x]

# Midpoints
MidPoints = []
i=0
for i in range(len(interval)):
    MidPoints.append((sum(interval[i]))/(len(interval[i])))

L_V1 = np.array([])
i=0
for i in range(len(interval)):
    L_V1 = np.append(L_V1,i+1)

FL1 = np.c_[L_V1,MidPoints]

i=0
j=0
FLRS = []
for i in range(len(data)):
    for j in range(len(interval)):
        if data[i] >= min(interval[j]) and data[i] <= max(interval[j]):
            FLRS.append(FL1[j][0])
            break

FLR = np.c_[data,FLRS]
FLR = FLR.tolist()

x= []
for i in range(len(FLRS)-1):
    x.append(FLRS[i+1])
    if i == len(FLRS)-1:
        break
   
x.append(0)
FLR_1 = np.c_[FLR,x]

FOLRS = []
i=0
j=0
for i in range(len(FLR)):
    temp = []
    for j in range(len(FLR_1)):
        if FLR_1[i][1] == FLR_1[j][1]:
            temp.append(FLR_1[j][2])
            A = [i for i in temp if i!=0]
    FOLRS.append(A)

FOLRS = np.array(FOLRS)

i=0
for i in range(len(FOLRS)):
	np.pad(FOLRS[i],(0,len(FOLRS[0])-len(FOLRS[i])), mode='constant')
	
midP = []

i=0
j=0
k=0

for i in range(len(FOLRS)):
    temp = []
    for j in range(len(FOLRS[i])):
        for k in range(len(FL1)):
            if FOLRS[i][j] == FL1[k][0]:
                temp.append(round(FL1[k][1],1))
    midP.append(temp)

out = 0
out1 = []
i=0
j=0

for i in range(len(FOLRS)):
    for j in range(len(FOLRS[i])):
        out = out + (midP[i][j])*((FOLRS[i][j])/sum(FOLRS[i]))

    out1.append(round(out,1))
    out = 0

plt.plot(data, label='Actual data')
plt.plot(out1, label='Predicted data')
plt.xlabel('Days')
plt.ylabel('number of cases')
plt.title('Cases per day')
plt.legend()
plt.show()

np.savetxt('output.csv',np.c_[data,out1], fmt="%d",delimiter=',')

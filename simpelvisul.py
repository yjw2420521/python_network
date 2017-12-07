#导入画图包
import matplotlib.pyplot as plt
#这是计算包
import numpy as np
#这是画图包
import pandas as pd
#添加中文包
from pylab import *  
mpl.rcParams['font.sans-serif'] = ['SimHei']  
file="data.csv"
df = pd.read_csv(file)
print(type(df))
#读取0到100的第四列
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa',-1,1)
# print(y)
#把0到100的第一行和第二行当做数组存入X中
X = df.iloc[0:100,[0,2]].values
#我在找第二行等于7的噪声
# for item in df.iloc[0:100,2].values:
#     i = 0 
#     i  += 0
#     if item == 7:
#         print(i)
#     else:
#         print('no')
#         pass


#第一个数据集为o点红色 第一列的数据为x轴 第二列的数据为y轴
plt.scatter(X[:50,0],X[:50,1],color='red',marker = 'o',label='setosa')
#第二个数据集为x点蓝色 同上
plt.scatter(X[50:100,0],X[50:100,1],color='blue',marker = 'x',label='versicolor')
plt.xlabel('花瓣的大小')
plt.ylabel('花径的长度')
plt.legend(loc='upper left')
print(X)
plt.show()


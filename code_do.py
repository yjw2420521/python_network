import simpelvisul
import ganzhiqi
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np

def plot_decision_regions(X,y,classifier,resolution=0.02):
    markers = ('s','x','o','v')
    colors = ('red','blue','lightgreen','gray','cray')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    """
    一下两种代码是获取花瓣长度和大小的最大最小值
    其中0列是花瓣的大小 1列是花径的长度 
    """
    x1_min,x1_max = X[:, 0].min() - 1,X[:,0].max()
    x2_min,x2_max = X[:, 1].min() - 1,X[:,1].max()
    print(x1_max,x1_min)
    print(x2_max,x2_min)
    """
    也就是说xx1输出的是一个从最小值到到最大值的一个数组
    np.meshgrid 就是变成二维数组
    """
    xx1,xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),np.arange(x2_min,x2_max,resolution))
    print(xx2)
    print(xx1)
    # print(xx1.shape)
    """
    xx1.ravel()就是还原向量到单维数据
    """
    Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    print(xx1.ravel())
    print(xx2.ravel())
    print(Z)
    Z = Z.reshape(xx1.shape)
    """
    contourf是用于对线进行分类 对于线性可分的模型神经计算会给我们画出一个分界线
    """
    plt.contourf(xx1,xx2,Z,alpha=0.4,cmap=cmap)
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())

    for idx,cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl,0],y = X[y == cl,1],alpha=0.8,c=cmap(idx),marker=markers[idx],label=cl)

# print(X)
X = simpelvisul.X
y = simpelvisul.y
ppn = ganzhiqi.Perceptron(eta=0.1,n_iter=10)
ppn.fit(X,y)
plt.plot(range(1,len(ppn.errors_)+1),ppn.errors_,marker='o')
plt.xlabel('Epochs')
plt.ylabel('错误分类次数')
# plt.show()
plot_decision_regions(X,y,ppn,resolution=0.02)
plt.xlabel('花径长度')
plt.ylabel('花瓣长度')
plt.legend(loc='upper left')
plt.show()
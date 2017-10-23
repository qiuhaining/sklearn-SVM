"""
==============
Non-linear SVM
==============

Perform binary classification using non-linear SVC
with RBF kernel. The target to predict is a XOR of the
inputs.

The color map illustrates the decision function learned by the SVC.
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

xx1, yy1 = np.meshgrid(np.linspace(-3, 3, 500),
                     np.linspace(-3, 3, 500))
np.random.seed(0)#用于生成相同的随机数，与时间无关
X = np.random.randn(300, 2)
#logical_xor进行逻辑或运算
Y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)
xx,yy=np.mgrid[X[:,0].min():X[:,0].max():300j,X[:,1].min():X[:,1].max():300j]
# fit the model
clf = svm.NuSVC()
clf.fit(X, Y)

# plot the decision function for each datapoint on the grid
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
#decision_function返回样本距超平面的距离w*x+b
Z = Z.reshape(xx.shape)
#Imshow：用颜色表示的二维图像,深浅代表大小,色彩平面图
plt.imshow(Z,interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto',
           origin='lower', cmap=plt.cm.PuOr_r)
#contour:levels=[0]:画出等高线为0的线，在这里就是超平面
contours = plt.contour(xx, yy, Z, levels=[0], linewidths=2,
                       linetypes='--')
plt.scatter(X[:, 0], X[:, 1], s=30, c=Y, cmap=plt.cm.Paired)
plt.xticks(())
plt.yticks(())
plt.axis([-3, 3, -3, 3])
plt.show()

# plot the decision function for each datapoint on the grid
Z1 = clf.decision_function(np.c_[xx1.ravel(), yy1.ravel()])
#decision_function返回样本距超平面的距离w*x+b
Z1 = Z1.reshape(xx1.shape)
#Imshow：用颜色表示的二维图像,深浅代表大小,色彩平面图
plt.imshow(Z1,interpolation='nearest',
           extent=(xx1.min(), xx1.max(), yy1.min(), yy1.max()), aspect='auto',
           origin='lower', cmap=plt.cm.PuOr_r)
#contour:levels=[0]:画出等高线为0的线，在这里就是超平面
contours = plt.contour(xx1, yy1, Z1, levels=[0], linewidths=2,
                       linetypes='--')
plt.scatter(X[:, 0], X[:, 1], s=30, c=Y, cmap=plt.cm.Paired)
plt.xticks(())
plt.yticks(())
plt.axis([-3, 3, -3, 3])
plt.show()

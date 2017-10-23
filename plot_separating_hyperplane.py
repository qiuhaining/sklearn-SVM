"""
=========================================
SVM: Maximum margin separating hyperplane
=========================================

Plot the maximum margin separating hyperplane within a two-class
separable dataset using a Support Vector Machine classifier with
linear kernel.
"""
#   ###能找到边界，超平面和支持向量
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# we create 40 separable points
np.random.seed(0)
X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
Y = [0] * 20 + [1] * 20#列表形式Type=list

# fit the model
clf = svm.SVC(kernel='linear')
clf.fit(X, Y)

# get the separating hyperplane
#在二维空间内
w = clf.coef_[0]#参数法向量w*
a = -w[0] / w[1]#超平面斜率
xx = np.linspace(-5, 5)
yy = a * xx - (clf.intercept_[0]) / w[1]#参数截距b，yy是超平面上第二维的坐标

# plot the parallels to the separating hyperplane that pass through the
# support vectors
b = clf.support_vectors_[0]#返回支持向量的坐标
yy_down = a * xx + (b[1] - a * b[0])
b = clf.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])

# plot the line, the points, and the nearest vectors to the plane
plt.plot(xx, yy, 'k-')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')

plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, -1],
            s=80, facecolors='none')
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)

plt.axis('tight')
plt.show()

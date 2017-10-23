"""
=================================================
SVM-Anova: SVM with univariate feature selection
=================================================

This example shows how to perform univariate feature selection before running a
SVC (support vector classifier) to improve the classification scores.
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets, feature_selection
from sklearn.cross_validation import cross_val_score#交叉验证
from sklearn.pipeline import Pipeline

###############################################################################
# Import some data to play with
digits = datasets.load_digits()
y = digits.target
# Throw away data, to be in the curse of dimension settings
y = y[:200]
X = digits.data[:200]
n_samples = len(y)
X = X.reshape((n_samples, -1))
# add 200 non-informative features
X = np.hstack((X, 2 * np.random.random((n_samples, 200))))

###############################################################################
# Create a feature-selection transform and an instance of SVM that we
# combine together to have an full-blown estimator
#feature_selection.f_classif：计算所提供样本的方差分析F-值anova：方差分析
#feature_selection.SelectPercentile（k）：只留下k值最高的一组特征，返回最终的估计器
transform = feature_selection.SelectPercentile(feature_selection.f_classif)
#anova：Analysis of Variance(方差分析)
#http://baike.baidu.com/link?url=8ufVQvD2KZrWbS3VvvuhYDfw3dk8nSD84QRUNB1P864
#rW8XKSw6-P4-xGIHVkAEBHUIjQGFhFsPtQhazMQrUVmcAqLVDBkQKVXSb3MPq92QFhPaPmVyEgsMNF
#ZJ_p1B-QyQ-tHMQKFJB_recu1qG9nDDpfdDbwMAomoktviOFca
clf = Pipeline([('anova', transform), ('svc', svm.SVC(C=1.0))])

###############################################################################
# Plot the cross-validation score as a function of percentile of features
score_means = list()
score_stds = list()
percentiles = (1, 3, 6, 10, 15, 20, 30, 40, 60, 80, 100)

for percentile in percentiles:
    #clf.set_params：设置此估计器的参数。
    #使用网格搜索(grid search)和交叉验证(cross validation)来选择参数.
    #clf.set_params(svm__C=1.0*percentile/100)
    #对方差分析中的参数percentile进行调节，实现多重比较检验
    #用于确定控制变量的不同水平对观测变量的影响程度如何
    clf.set_params(anova__percentile=percentile)
    # Compute cross-validation score using 1 CPU
    #http://scikit-learn.org/dev/modules/generated/sklearn.model_selection.
    #cross_val_score.html#sklearn.model_selection.cross_val_score
    #cross_val_score：最简单的交叉验证方法，cv选择折数，默认是3折交叉验证
    this_scores = cross_val_score(clf, X, y, n_jobs=1)
    score_means.append(this_scores.mean())
    score_stds.append(this_scores.std())
#plt.errorbar以折线形式画出均值和方差
plt.errorbar(percentiles, score_means, np.array(score_stds))

plt.title(
    'Performance of the SVM-Anova varying the percentile of features selected')
plt.xlabel('Percentile')
plt.ylabel('Prediction rate')

plt.axis('tight')
plt.show()

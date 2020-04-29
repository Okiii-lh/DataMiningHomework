# coding=utf-8
"""
@File    :   Analyze.py    
@Contact :   13132515202@163.com

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2020/4/28 20:12   LiuHe      1.0         None
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pylab import mpl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


def plot_roc_curve(fprs,tprs):
    plt.figure(figsize=(8,6),dpi=80)
    plt.plot(fprs,tprs)
    plt.plot([0,1],linestyle='--')
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.ylabel('TP rate',fontsize=15)
    plt.xlabel('FP rate',fontsize=15)
    plt.title('ROC曲线',fontsize=17)
    plt.show()



mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

df = pd.read_csv("../data/heart.csv")

print(df.info())

print(df.target.value_counts())

sns.countplot(x='target', data=df, palette="muted")
plt.xlabel("得病/未得病比例")
plt.show()

print(df.describe())

first = pd.get_dummies(df['cp'], prefix="cp")
second = pd.get_dummies(df['slope'], prefix="slope")
thrid = pd.get_dummies(df['thal'], prefix="thal")

df = pd.concat([df, first, second, thrid], axis=1)
df = df.drop(columns=['cp', 'slope', 'thal'])
print(df.head(3))


y = df.target.values
X = df.drop(['target'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=6)

standardScaler = StandardScaler()
standardScaler.fit(X_train)
X_train = standardScaler.transform(X_train)
X_test = standardScaler.transform(X_test)


rf_clf = RandomForestClassifier(n_estimators=500, random_state=666, oob_score=True, n_jobs=-1)
rf_clf.fit(X, y)

RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=500, n_jobs=-1,
            oob_score=True, random_state=666, verbose=0, warm_start=False)


print(rf_clf.oob_score_)

print(rf_clf.score(X_test, y_test))

y_probabilities_rf = rf_clf.predict_proba(X_test)[:, 1]

print(roc_auc_score(y_test, y_probabilities_rf))

y_probabilities = rf_clf.predict_proba(X)[:, 1]

print(roc_auc_score(y, y_probabilities))

fprs4, tprs4, thresholds4 = roc_curve(y_test, y_probabilities_rf)
# 此处调用前面的绘制函数
plot_roc_curve(fprs4, tprs4)


rf_clf2 = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, random_state=666, oob_score=True, n_jobs=-1)
rf_clf2.fit(X,y)

RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=16,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=500, n_jobs=-1,
            oob_score=True, random_state=666, verbose=0, warm_start=False)

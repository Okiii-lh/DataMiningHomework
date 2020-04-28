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

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

df = pd.read_csv("../data/heart.csv")

print(df.info())

print(df.target.value_counts())

sns.countplot(x='target', data=df, palette="muted")
plt.xlabel("得病/未得病比例")
plt.show()
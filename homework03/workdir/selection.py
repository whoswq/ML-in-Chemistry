"""
author: Wang Chongbin
通过计算Pearson相关系数的方法来选择学习模型感兴趣的特征
"""

import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np
import pandas as pd

df_train = pd.read_csv("./data/train.csv")


def transform(df_train):
    """
    将原始数据中的特征替换为数值
    删去一些不用的特征
    """
    df_train = df_train.drop(["Unnamed: 0", "id", "Driving_License"], axis=1)
    df_train = df_train.replace({"Gender": {"Male": 1, "Female": 0}})
    df_train = df_train.replace(
        {"Vehicle_Age": {
            "< 1 Year": 0,
            "1-2 Year": 1,
            "> 2 Years": 2
        }})
    df_train = df_train.replace({"Vehicle_Damage": {"Yes": 1, "No": 0}})

    # 使用min-max方法归一化数据
    def min_max(feature: str):
        min_ = df_train[feature].min()
        max_ = df_train[feature].max()
        df_train[feature] = df_train[feature].map(lambda x: (x - min_) /
                                                  (max_ - min_))

    df_train["Vehicle_Age"] = df_train["Vehicle_Age"].div(2)
    min_max("Age")
    min_max("Region_Code")
    min_max("Annual_Premium")
    min_max("Policy_Sales_Channel")
    min_max("Vintage")
    return df_train


df_train = transform(df_train)

# 计算Pearson系数并作图
corrmat = np.ones((10, 10))
feature_list = [
    "Gender", "Age", "Region_Code", "Previously_Insured", "Vehicle_Age",
    "Vehicle_Damage", "Annual_Premium", "Policy_Sales_Channel", "Vintage",
    "Response"
]
for i in range(10):
    for j in range(i + 1, 10):
        corrmat[i][j] = corrmat[j][i] = abs(
            pearsonr(df_train[feature_list[i]], df_train[feature_list[j]])[0])
plt.figure()
plt.matshow(corrmat)
plt.colorbar()
plt.savefig('./workdir/pearson_corr.png', dpi=800)
plt.show()

"""
author: Wang Chongbin
绘制作业第二题中的图
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df_train = pd.read_csv("data/train_valid.csv", header=0)
# df_test = pd.read_csv("data/test.csv", header=0)

# 首先分析年龄与购买保险倾向之间的关系
# 提取Age与Response字段的数据
response = df_train["Response"].values
age = df_train["Age"].values
age_P = age[response == 1]
age_N = age[response == 0]
plt.figure()
plt.boxplot([age_P, age_N], labels=["Positive", "Negative"])
plt.title("Relation Between Response and Age")
plt.xlabel("Data Labels")
plt.ylabel("Age")
plt.savefig("./workdir/age_boxplot.png", dpi=800)
plt.close()


# 分析保险年费与购买倾向之间的关系
# 提取Annual_Premium字段的数据
cost = df_train["Annual_Premium"].values
cost_P = cost[response == 1]
cost_N = cost[response == 0]
plt.figure()
plt.boxplot([cost_P, cost_N], labels=["Positive", "Negative"])
plt.title("Relation Between Response and Annual Premium")
plt.xlabel("Data Labels")
plt.ylabel("Annual Premium")
plt.savefig("./workdir/annual_premium_boxplot.png", dpi=800)
plt.close()
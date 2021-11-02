"""
author: Wang Chongbin
将训练集划分为训练集和验证集
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df_train = pd.read_csv("data/train_valid.csv", header=0)

df_train, df_valid = train_test_split(df_train)
ratio = df_train["id"].count() / df_valid["id"].count()
df_train.to_csv("./data/train.csv")
df_valid.to_csv("./data/valid.csv")

import numpy as np
import pandas as pd

# 读取数据
df_train = pd.read_csv("data/train.csv", header=0)
df_valid = pd.read_csv("data/valid.csv", header=0)
# 下面计算题目中需要的数据
# 训练集和数据集中的男女比例
gender_ratio_train = sum(df_train["Gender"].values == "Male") / sum(
    df_train["Gender"].values == "Female")
gender_ratio_valid = sum(df_valid["Gender"].values == "Male") / sum(
    df_valid["Gender"].values == "Female")
print("gender ratio in train data is %.3f" % gender_ratio_train)
print("gender ratio in valid data is %.3f" % gender_ratio_valid)
# 年龄
print("maximium age in train data is %d" % df_train["Age"].max())
print("minimium age in train data is %d" % df_train["Age"].min())
print("average age of train data is %.2f" % df_train["Age"].mean())
print("median age of train data is %.1f" % df_train["Age"].median())
print("maximium age in valid data is %d" % df_valid["Age"].max())
print("minimium age in valid data is %d" % df_valid["Age"].min())
print("average age of valid data is %.2f" % df_valid["Age"].mean())
print("median age of valid data is %.1f" % df_valid["Age"].median())

# 驾照比例
# 首先提取Driving_License字段的数据
drv_lcs_train = df_train["Driving_License"].value_counts()
drv_lcs_valid = df_valid["Driving_License"].value_counts()
drv_lcs_ratio_train = drv_lcs_train[1] / (drv_lcs_train[0] + drv_lcs_train[1])
drv_lsc_ratio_valid = drv_lcs_valid[1] / (drv_lcs_valid[0] + drv_lcs_valid[1])
print("driving license ratio in train data is %.4f" % drv_lcs_ratio_train)
print("driving license ratio in valid data is %.4f" % drv_lsc_ratio_valid)

# 之前购买保险的比例
# 首先提取Previously_Insured字段的数据
train_prv_ins = df_train["Previously_Insured"].value_counts()
valid_prv_ins = df_valid["Previously_Insured"].value_counts()
prv_ins_train_ratio = train_prv_ins[1] / (train_prv_ins[1] + train_prv_ins[0])
prv_ins_valid_ratio = valid_prv_ins[1] / (valid_prv_ins[1] + valid_prv_ins[0])
print("previously insured ratio in train data is %.4f" % prv_ins_train_ratio)
print("previously insured ratio in valid data is %.4f" % prv_ins_valid_ratio)

# 汽车年限比例
# 提取Vehicle_Age字段的数据
v_age_train = df_train["Vehicle_Age"].value_counts()
v_age_valid = df_valid["Vehicle_Age"].value_counts()
print("< 1 year ratio in train data is %.4f" %
      (v_age_train["< 1 Year"] / df_train["Vehicle_Age"].count()))
print("1-2 year ratio in train data is %.4f" %
      (v_age_train["1-2 Year"] / df_train["Vehicle_Age"].count()))
print(">2 years ratio in train data is %.4f" %
      (v_age_train["> 2 Years"] / df_train["Vehicle_Age"].count()))
print("< 1 year ratio in valid data is %.4f" %
      (v_age_valid["< 1 Year"] / df_valid["Vehicle_Age"].count()))
print("1-2 year ratio in valid data is %.4f" %
      (v_age_valid["1-2 Year"] / df_valid["Vehicle_Age"].count()))
print(">2 years ratio in valid data is %.4f" %
      (v_age_valid["> 2 Years"] / df_valid["Vehicle_Age"].count()))

# 汽车曾经损坏的比例
# 提取Vehicle_Damage字段的数据
damage_train = df_train["Vehicle_Damage"].value_counts()
damage_valid = df_valid["Vehicle_Damage"].value_counts()
print("damaged ration in train data is %.4f" %
      (damage_train["Yes"] / df_train["Vehicle_Damage"].count()))
print("damaged ration in valid data is %.4f" %
      (damage_valid["Yes"] / df_valid["Vehicle_Damage"].count()))

# 年保险费
# 提取Annual_Premium字段的数据
print("maximium of annual ptrmium in train data is %.2f" %
      df_train["Annual_Premium"].max())
print("minimium of annual ptrmium in train data is %.2f" %
      df_train["Annual_Premium"].min())
print("average of annual ptrmium in train data is %.2f" %
      df_train["Annual_Premium"].mean())
print("median of annual ptrmium in train data is %.2f" %
      df_train["Annual_Premium"].median())
print("maximium of annual ptrmium in valid data is %.2f" %
      df_valid["Annual_Premium"].max())
print("minimium of annual ptrmium in valid data is %.2f" %
      df_valid["Annual_Premium"].min())
print("average of annual ptrmium in valid data is %.2f" %
      df_valid["Annual_Premium"].mean())
print("median of annual ptrmium in valid data is %.2f" %
      df_valid["Annual_Premium"].median())

"""
author: Wang Chongbin
绘制作业第一题中的图
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df_train = pd.read_csv("data/train_valid.csv", header=0)
# df_test = pd.read_csv("data/test.csv", header=0)

# 首先绘制性别与保险购买倾向的关系
# 将Gender字段的数据改为int，Male->1, Female->0
df_train = df_train.replace({"Gender": {
    "Male": 1,
    "Female": 0
}})  # 竟然不是修改原dateframe是否有点反人类
# 提取Gender与Respose的数据
gender = df_train["Gender"].values
response = df_train["Response"].values
# 首先计算男女比例
total_number = len(gender)
male_index = (gender == 1)
female_index = (gender == 0)
male_number = len(gender[male_index])
female_number = len(gender[female_index])
male_ratio = male_number / total_number
female_ratio = female_number / total_number
# 现在计算每种性别购买保险的意愿
male_P_ratio = len(
    response[male_index][response[male_index] == 1]) / male_number
male_N_ratio = len(
    response[male_index][response[male_index] == 0]) / male_number
female_P_ratio = sum(response[female_index] == 1) / female_number
female_N_ratio = len(
    response[female_index][response[female_index] == 0]) / female_number
# 绘图
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.bar([0.5, 1.5], [male_N_ratio, female_N_ratio],
        0.6,
        0.,
        color="steelblue",
        label="Negative Ratio")
ax1.bar([0.5, 1.5], [male_P_ratio, female_P_ratio],
        0.6, [male_N_ratio, female_N_ratio],
        color="darkorange",
        label="Positive Ratio")
ax1.set_title("Preference of Different Gender")
ax1.set_xticks([0.5, 1.5])
ax1.set_xticklabels(["Male", "Female"])
ax1.set_ylabel("Percentage")
ax1.legend(bbox_to_anchor=(.98, 1.02))

ax2.pie([male_ratio, female_ratio],
        labels=["Male", "Female"],
        autopct="%.2f%%",
        startangle=90)
ax2.set_title("Gender Ratio")
ax2.axis("equal")
plt.savefig("workdir//1-gender-resonse.png", dpi=800)
print("workdir//1-gender-resonse.png has been saved")
plt.close()

# 绘制汽车年限与购买保险倾向之间的关系图
# 将Vehicle_Age字段的数据转化为int
df_train = df_train.replace(
    {"Vehicle_Age": {
        "< 1 Year": 0,
        "1-2 Year": 1,
        "> 2 Years": 2
    }})
# 提取Vehicle_Age的原始数据
vehicle_age = df_train["Vehicle_Age"].values
# 计算不同年限的车辆的比例
age_0_index = (vehicle_age == 0)
age_0_number = sum(age_0_index)
age_0_ratio = age_0_number / total_number
age_1_index = (vehicle_age == 1)
age_1_number = sum(age_1_index)
age_1_ratio = age_1_number / total_number
age_2_index = (vehicle_age == 2)
age_2_number = sum(age_2_index)
age_2_ratio = age_2_number / total_number
# 计算不同年限车辆车主的购买保险意愿
age_0_P_ratio = sum(response[age_0_index] == 1) / age_0_number
age_0_N_ratio = sum(response[age_0_index] == 0) / age_0_number
age_1_P_ratio = sum(response[age_1_index] == 1) / age_1_number
age_1_N_ratio = sum(response[age_1_index] == 0) / age_1_number
age_2_P_ratio = sum(response[age_2_index] == 1) / age_2_number
age_2_N_ratio = sum(response[age_2_index] == 0) / age_2_number
# 绘图

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.bar([0.5, 1.5, 2.5], [age_0_N_ratio, age_1_N_ratio, age_2_N_ratio],
        0.6,
        0.,
        color="steelblue",
        label="Negative Ratio")
ax1.bar([0.5, 1.5, 2.5], [age_0_P_ratio, age_1_P_ratio, age_2_P_ratio],
        0.6, [age_0_N_ratio, age_1_N_ratio, age_2_N_ratio],
        color="darkorange",
        label="Positive Ratio")
ax1.set_title("Preference of Different Vehicle Age")
ax1.set_xticks([0.5, 1.5, 2.5])
ax1.set_xticklabels(["<1 year", "1-2 year", ">2 years"])
ax1.set_ylabel("Percentage")
ax1.legend(bbox_to_anchor=(.98, 1.02))

ax2.pie([age_0_ratio, age_1_ratio, age_2_ratio],
        labels=["<1 year", "1-2 year", ">2 years"],
        autopct="%.2f%%",
        startangle=90)
ax2.set_title("Vehicle Age Ratio")
ax2.axis("equal")
plt.savefig("workdir//1-vehicle-age-resonse.png", dpi=800)
print("workdir//1-vehicle-age-resonse.png has been saved")
plt.close()

# 绘制之前是否购买过保险与购买保险意愿之间的关系图
# 提取Previously_Insured字段的数据
previously_insured = df_train["Previously_Insured"].values
# 计算曾经买过保险的人数比例
pre_insured_index = (previously_insured == 1)
pre_insured_number = sum(pre_insured_index)
pre_uninsured_index = (previously_insured == 0)
pre_uninsured_number = sum(pre_uninsured_index)
pre_insured_ratio = pre_insured_number / total_number
pre_uninsured_ratio = pre_uninsured_number / total_number
# 分析曾经买过保险与买保险意愿之间的关系
pre_insured_P_ratio = sum(
    response[pre_insured_index] == 1) / pre_insured_number
pre_insured_N_ratio = sum(
    response[pre_insured_index] == 0) / pre_insured_number
pre_uninsured_P_ratio = sum(
    response[pre_uninsured_index] == 1) / pre_uninsured_number
pre_uninsured_N_ratio = sum(
    response[pre_uninsured_index] == 0) / pre_uninsured_number
# 绘图
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.bar([0.5, 1.5], [pre_insured_N_ratio, pre_uninsured_N_ratio],
        0.6,
        0.,
        color="steelblue",
        label="Negative Ratio")
ax1.bar([0.5, 1.5], [pre_insured_P_ratio, pre_uninsured_P_ratio],
        0.6, [pre_insured_N_ratio, pre_uninsured_N_ratio],
        color="darkorange",
        label="Positive Ratio")
ax1.set_title("Preference of Different Insured Users")
ax1.set_xticks([0.5, 1.5])
ax1.set_xticklabels(["insured", "uninsured"])
ax1.set_ylabel("Percentage")
ax1.legend(bbox_to_anchor=(.98, 1.02))

ax2.pie([pre_insured_ratio, pre_uninsured_ratio],
        labels=["insured", "uninsured"],
        autopct="%.2f%%",
        startangle=90)
ax2.set_title("Different Insured  Ratio")
ax2.axis("equal")
plt.savefig("workdir//1-pre-ins-resonse.png", dpi=800)
print("workdir//1-pre-ins-resonse.png has been saved")
plt.close()

# 分析汽车是否曾损坏与购买保险意愿之间的关系
# 将Vehicle_Damage字段的数据转化为int
df_train = df_train.replace({"Vehicle_Damage": {"Yes": 1, "No": 0}})
# 提取Vehicle_Damage字段的数据
damage = df_train["Vehicle_Damage"].values
# 首先计算车辆曾经损坏所占的人数比例
Y_damage_index = (damage == 1)
N_damage_index = (damage == 0)
Y_damage_number = sum(Y_damage_index)
N_damage_number = sum(N_damage_index)
Y_damage_ratio = Y_damage_number / total_number
N_damage_ratio = N_damage_number / total_number
# 计算不同车辆损坏情况的用户的购买保险倾向
Y_damage_P_ratio = sum(response[Y_damage_index] == 1) / Y_damage_number
Y_damage_N_ratio = sum(response[Y_damage_index] == 0) / Y_damage_number
N_damage_P_ratio = sum(response[N_damage_index] == 1) / N_damage_number
N_damage_N_ratio = sum(response[N_damage_index] == 0) / N_damage_number
# 绘图
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.bar([0.5, 1.5], [N_damage_N_ratio, Y_damage_N_ratio],
        0.6,
        0.,
        color="steelblue",
        label="Negative Ratio")
ax1.bar([0.5, 1.5], [N_damage_P_ratio, Y_damage_P_ratio],
        0.6,  [N_damage_N_ratio, Y_damage_N_ratio],
        color="darkorange",
        label="Positive Ratio")
ax1.set_title("Preference of Different Damaged Users")
ax1.set_xticks([0.5, 1.5])
ax1.set_xticklabels(["Undamaged", "Damaged"])
ax1.set_ylabel("Percentage")
ax1.legend(bbox_to_anchor=(.98, 1.02))

ax2.pie([Y_damage_ratio, N_damage_ratio],
        labels=["Damaged", "Undamaged"],
        autopct="%.2f%%",
        startangle=90)
ax2.set_title("Different Insured Ratio")
ax2.axis("equal")
plt.savefig("workdir//1-damaged-resonse.png", dpi=800)
print("workdir//1-damaged-resonse.png has been saved")
plt.show()
plt.close()
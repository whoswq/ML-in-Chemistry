#!/usr/bin/env python
#coding=utf-8
"""
Author:        Fanchong Jian, Pengbo Song
Created Date:  2021/10/15
Last Modified: 2021/10/17
"""

from warnings import warn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model, metrics


class MLChemLab2(object):
    """Template class for Lab2 -*- Logistic Regression -*-

    Properties:
        model: Logistic model to fit.
        featurization_mode: Keyword to choose feature methods.
    """
    def __init__(self):
        """Initialize class with empty model and default featurization mode to identical"""
        self.model = None
        self.featurization_mode = "identical"

    def fit(self, X, y, featurization_mode: str = "normalization"):
        """Feature input X using given mode and fit model with featurized X and y
        
        Args:
            X: Input X data.
            y: Input y data.
            featurization_mode: Keyword to choose feature methods.

        Returns:
            Trained model using given X and y, or None if no model is specified
        """
        # Catch empty model, a model should be added earlier
        if (self.model is None):
            warn("No model to fit. Nothing Returned.")
            return None
        # Set featurization mode keyword
        self.featurization_mode = featurization_mode
        # Preprocess X
        featurized_X = self.featurization(X)
        self.model.fit(featurized_X, y)
        return self.model

    def add_model(self, kw: str = "logistic", **kwargs):
        """Add model before fitting and prediction

        Args:
            kw: Keyword that indicates which model to build.
            kwargs: Keyword arguments passed to 
        """
        if kw == "logistic":
            self.model = linear_model.LogisticRegression(**kwargs)
        else:
            # Catch incorrect keywords
            raise NotImplementedError("Got incorrect model keyword " + kw)

    def featurization(self, X):
        """Feature input X data using preset mode"""
        def min_max(vec):
            return (vec - vec.min()) / (vec.max() - vec.min())

        if self.featurization_mode == "normalization":
            n_f = len(X[0])
            n_s = len(X)
            X_norm = np.zeros((n_f, n_s))
            for i in range(n_f):
                X_norm[i] = min_max(X[..., i])
            return X_norm.T
        elif self.featurization_mode == "identical":
            return X

    def predict(self, X):
        """Predict based on fitted model and given X data"""
        # Catch empty model, a model should be added earlier
        if (self.model is None):
            warn("No model to fit. Nothing Returned.")
            return None
        X = self.featurization(X)
        y_pred = self.model.predict(X)
        return y_pred

    def evaluation(self, y_true, y_pred, metric: str = "accuracy"):
        """Eavluate training results based on predicted y and true y"""
        if metric == "accuracy":
            return metrics.accuracy_score(y_true, y_pred)
        elif metric == "precision":
            return metrics.precision_score(y_true, y_pred)
        elif metric == "recall":
            return metrics.recall_score(y_true, y_pred)
        elif metric == "F1":
            return metrics.f1_score(y_true, y_pred)
        elif metric == "CM":
            return metrics.confusion_matrix(y_true, y_pred)
        elif metric == "AUC":
            return metrics.roc_auc_score(y_true, y_pred)
        else:
            raise NotImplementedError("Got incorrect metric keyword " + metric)


def transform(df_train):
    """
    将原始数据中的特征替换为数值
    """
    """df_train = df_train.drop([
        "Unnamed: 0", "id", "Driving_License", "Region_Code", "Annual_Premium",
        "Vintage"
    ],
                             axis=1)"""
    df_train = df_train.replace({"Gender": {"Male": 1, "Female": 0}})
    df_train = df_train.replace(
        {"Vehicle_Age": {
            "< 1 Year": 0,
            "1-2 Year": 1,
            "> 2 Years": 2
        }})
    df_train = df_train.replace({"Vehicle_Damage": {"Yes": 1, "No": 0}})

    # 使用min-max方法归一化数据
    """def min_max(feature: str):
        min_ = df_train[feature].min()
        max_ = df_train[feature].max()
        df_train[feature] = df_train[feature].map(lambda x: (x - min_) /
                                                  (max_ - min_))

    df_train["Vehicle_Age"] = df_train["Vehicle_Age"].div(2)
    min_max("Age")
    min_max("Region_Code")
    min_max("Annual_Premium")
    min_max("Policy_Sales_Channel")
    min_max("Vintage")"""
    return df_train


def main():
    """General workflow of machine learning
    1. Prepare dataset
    2. Build model
    3. Data preprocessing (featurization, normalization, ...)
    4. Training
    5. Predict
    6. Model evalution
    """

    # Step 1: Prepare dataset
    df_train = pd.read_csv("./data/train.csv", header=0)
    df_test = pd.read_csv("./data/test.csv", header=0)
    df_valid = pd.read_csv("./data/valid.csv")
    id_train = list(df_train["id"].values)
    id_valid = list(df_valid["id"].values)
    id_test = list(df_test["id"].values)
    # 根据selection.py文件中的结论，只选择部分特征
    df_X_train = transform(df_train[[
        "Gender", "Age", "Previously_Insured", "Vehicle_Age", "Vehicle_Damage",
        "Policy_Sales_Channel"
    ]])
    df_y_train = df_train["Response"]
    df_X_valid = transform(df_valid[[
        "Gender", "Age", "Previously_Insured", "Vehicle_Age", "Vehicle_Damage",
        "Policy_Sales_Channel"
    ]])
    df_y_valid = df_valid["Response"]
    # Split valid dataset
    df_X_test = transform(df_test[[
        "Gender", "Age", "Previously_Insured", "Vehicle_Age", "Vehicle_Damage",
        "Policy_Sales_Channel"
    ]])

    # Step 2: Build model
    my_model = MLChemLab2()  # Instantiation of the custom class
    # 正则化参数在这里添加

    # Step 3: Data preprocessing
    # 将dataframe转换为numpy数组
    X_train = df_X_train.values
    y_train = df_y_train.values
    X_valid = df_X_valid.values
    y_valid = df_y_valid.values
    X_test = df_X_test.values

    # Step 4: Training
    """# 筛选出一个比较好的正则化参数，这段代码跑完大概需要20min
    ln_C = np.linspace(-20, -5, 1000)
    C_list = np.exp(ln_C)
    acc_train_list = []
    precision_train_list = []
    recall_train_list = []
    f1_train_list = []
    acc_valid_list = []
    precision_valid_list = []
    recall_valid_list = []
    f1_valid_list = []
    cnt = 0
    for C in C_list:
        my_model.add_model("logistic",
                           penalty="l2",
                           C=C,
                           class_weight="balanced")
        my_model.fit(X_train, y_train, featurization_mode="normalization"
                     )  # Fit model with the train dataset

        # Step 5: Predict
        y_train_pred = my_model.predict(
            X_train)  # Make prediction using the trained model

        # Step 6: Model evalution
        acc_train = my_model.evaluation(
            y_train, y_train_pred,
            metric="accuracy")  # Model evaluation with the test dataset
        precision_train = my_model.evaluation(y_train,
                                              y_train_pred,
                                              metric="precision")
        recall_train = my_model.evaluation(y_train,
                                           y_train_pred,
                                           metric="recall")
        f1_train = my_model.evaluation(y_train, y_train_pred, metric="F1")

        acc_train_list.append(acc_train)
        precision_train_list.append(precision_train)
        recall_train_list.append(recall_train)
        f1_train_list.append(f1_train)

        # 在验证集上的表现
        y_valid_pred = my_model.predict(
            X_valid)  # Make prediction using the trained model
        # Step 6: Model evalution
        acc_valid = my_model.evaluation(
            y_valid, y_valid_pred,
            metric="accuracy")  # Model evaluation with the test dataset
        precision_valid = my_model.evaluation(y_valid,
                                              y_valid_pred,
                                              metric="precision")
        recall_valid = my_model.evaluation(y_valid,
                                           y_valid_pred,
                                           metric="recall")
        f1_valid = my_model.evaluation(y_valid, y_valid_pred, metric="F1")

        acc_valid_list.append(acc_valid)
        precision_valid_list.append(precision_valid)
        recall_valid_list.append(recall_valid)
        f1_valid_list.append(f1_valid)
        print(cnt)
        cnt += 1
    figure, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(ln_C, acc_train_list, label="train acc")
    ax1.plot(ln_C, precision_train_list, label="train precision")
    ax1.plot(ln_C, recall_train_list, label="train recall")
    ax1.plot(ln_C, f1_train_list, label="train f1")
    ax2.plot(ln_C, acc_valid_list, label="valid acc")
    ax2.plot(ln_C, precision_valid_list, label="valid precision")
    ax2.plot(ln_C, recall_valid_list, label="valid recall")
    ax2.plot(ln_C, f1_valid_list, label="valid f1")
    ax1.set_xlabel("ln C")
    ax2.set_xlabel("ln C")
    ax1.legend()
    ax2.legend()
    plt.savefig("./workdir/selection_C.png", dpi=800)
    plt.show()"""

    my_model.add_model("logistic",
                       penalty="l2",
                       C=np.exp(-15),
                       class_weight="balanced")
    my_model.fit(X_train, y_train, featurization_mode="normalization")

    # Step 5: Predict
    y_train_pred = my_model.predict(
        X_train)  # Make prediction using the trained model

    # Step 6: Model evalution
    # 在训练集上的表现
    acc_train = my_model.evaluation(y_train, y_train_pred, metric="accuracy")
    precision_train = my_model.evaluation(y_train,
                                          y_train_pred,
                                          metric="precision")
    recall_train = my_model.evaluation(y_train, y_train_pred, metric="recall")
    f1_train = my_model.evaluation(y_train, y_train_pred, metric="F1")

    # 在验证集上的表现
    y_valid_pred = my_model.predict(X_valid)
    acc_valid = my_model.evaluation(
        y_valid, y_valid_pred,
        metric="accuracy")  # Model evaluation with the test dataset
    precision_valid = my_model.evaluation(y_valid,
                                          y_valid_pred,
                                          metric="precision")
    recall_valid = my_model.evaluation(y_valid, y_valid_pred, metric="recall")
    f1_valid = my_model.evaluation(y_valid, y_valid_pred, metric="F1")

    # 预测测试集的结果
    y_test_pred = my_model.predict(X_test)

    # 保存在测试集和训练集上的预测结果
    train_result_array = np.array([id_train, y_train_pred])
    df_train_res = pd.DataFrame(train_result_array.T,
                                columns=["id", "Response"])
    df_train_res.to_csv("./data/train_predict.csv")
    test_result_array = np.array([id_test, y_test_pred])
    df_test_res = pd.DataFrame(test_result_array.T, columns=["id", "Response"])
    df_test_res.to_csv("./data/test_predict.csv")
    valid_result_array = np.array([id_valid, y_valid_pred])
    df_valid_res = pd.DataFrame(valid_result_array.T,
                                columns=["id", "Response"])
    df_valid_res.to_csv("./data/valid_predict.csv")

    # 计算在训练集和验证集上的模型准确率
    print(f"ACCURACY_train = {acc_train:>.4f}")
    print(f"PRECISION_train = {precision_train:>.4f}")
    print(f"RECALL_train = {recall_train:>.4f}")
    print(f"F1_train = {f1_train:>.4f}")
    print(f"ACCURACY_valid = {acc_valid:>.4f}")
    print(f"PRECISION_valid = {precision_valid:>.4f}")
    print(f"RECALL_valid = {recall_valid:>.4f}")
    print(f"F1_valid = {f1_valid:>.4f}")
    # 计算训练集和验证集上的模型参数
    # 混淆矩阵
    cm_train = my_model.evaluation(y_train, y_train_pred, "CM")
    cm_valid = my_model.evaluation(y_valid, y_valid_pred, "CM")
    AUC_train = my_model.evaluation(y_train, y_train_pred, "AUC")
    AUC_valid = my_model.evaluation(y_valid, y_valid_pred, "AUC")
    print("confusion matrix of train data is:")
    print(cm_train)
    print("confusion matrix of valid data is:")
    print(cm_valid)
    print("AUC of train data is %.4f" % AUC_train)
    print("AUC of valid data is %.4f" % AUC_valid)


if __name__ == "__main__":
    main()

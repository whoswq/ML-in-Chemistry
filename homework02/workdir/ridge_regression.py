# coding=utf-8
"""
Author:         Chongbin Wang
Created Date:   2021/10/7
Last Modified:  2021/10/
"""

from warnings import warn
from sklearn import linear_model
import os
import numpy as np
import matplotlib.pyplot as plt


class MLChemLab1:
    """
    Lab1 Linear Regression

    Properties:
        model: linear model to fit [lasso, ridge, naive]
        featurization_mode:  keywords to choose feature methods
    """
    def __init__(self):
        """
        initialize class with empty model and default featurization mode
        to identical
        """
        self.model = None
        self.featurization_mode = "identical"

    def __str__(self):
        if self.model is None:
            return ("model: None\n please use '.add_mode()' to choose " /
                    "a type from [lasso, ridge, naive]")
        elif hasattr(self, "degree"):
            # this means that self.fit() has been uesd
            return "modle: %s\n featurization_mode: %s\n coefficients: %s" % (
                self.__model_name, self.featurization_mode, self.getparams())
        else:
            # this means that self.fit() has not been used
            return "modle: %s\n please use '.fit()' to train your model" \
                    % self.__model_name

    def add_mode(self, model: str, **kwargs):
        """
        set model before fitting and prediction

        model: str linear model
        **kwargs:
            alpha: double hyper parameter
        """
        if model == "ridge":
            self.__model_name = model
            if 'alpha' in kwargs:
                self.model = linear_model.Ridge(alpha=kwargs['alpha'])
            else:
                warn("Hyperparameter alpha for Ridge Regression not found!" /
                     "Using default alpha=0.5")
                self.model = linear_model.Ridge(alpha=0.5)

        elif model == "lasso":
            # Put your Lasso Regression model HERE
            self.model = None
            self.__model_name = model
            warn("Lasso has not been implemented yet.")

        elif model == "naive":
            self.__model_name = model
            self.model = linear_model.LinearRegression()
        else:
            raise NotImplementedError("Got incorrect model keyword " + model)

    def fit(self, x, y, f_mode: str, degree=None):
        """Feature input x using given mode and fit model with featurized X and y

        Args:
            x: Input x data.
            y: Input y data.
            f_mode: Keyword to choose feature methods.
            degree: the degree of polynomials

        Returns:
            Trained model using given x and y, or None if no model is specified
        """

        if (self.model is None):
            warn("No model to fit. Use 'self.add_mode()' first.")
            return None
        self.featurization_mode = f_mode
        self.degree = degree
        featurized_x = self.featurization(x, degree=degree)
        self.model.fit(featurized_x, y)
        return self.model

    def featurization(self, x, **kwargs):
        """Feature input X data using preset mode

        private function, there is no need to call this from outside
        """
        if self.featurization_mode == "poly":
            if 'degree' in kwargs and kwargs['degree'] is not None:
                degree = kwargs['degree']
            else:
                warn("Using default degree=4 for polynomial featurization")
                degree = 4  # default quartic polynomial
            X = np.power(x.reshape(-1, 1), np.linspace(
                1, degree,
                degree))  # get polynomial features [[X], [X^2], [X^3], ... ]
            return X

        elif self.featurization_mode == "poly-cos":
            if 'degree' in kwargs and kwargs['degree'] is not None:
                degree = kwargs['degree']
            else:
                warn("Using default degree=4 for polynomial featurization")
                degree = 4  # default quartic polynomial
            X = np.power(np.cos(x.reshape(-1, 1)),
                         np.linspace(1, degree, degree)
                         )  # get polynomial features [[X], [X^2], [X^3], ... ]
            return X
        elif self.featurization_mode == "identical":
            # Do nothing, returns raw X data
            return x

    def predict(self, x):
        """Predict based on fitted model and given X data"""
        X = self.featurization(x,
                               featurization_mode=self.featurization_mode,
                               degree=self.degree)
        y_predict = self.model.predict(X)
        return y_predict

    def evaluation(self, y_predict, y_label, metric: str):
        """Eavluate training results based on predicted y and true y"""
        if metric == "RMSE":
            return np.sqrt(
                np.sum(
                    np.power(y_predict.reshape(-1) - y_label.reshape(-1), 2.0))
                / y_predict.size)
        else:
            raise NotImplementedError("Got incorrect metric keyword " + metric)


def parse_dataset(train_data_file, test_data_file):
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    with open(train_data_file, "r") as f:  # open the train data file
        line = f.readline().strip()
        # read one line from the file and drop blank characters at the end
        while line:  # read until EOF
            line = line.split(' ')  # split the string with space
            train_x.append(float(line[0]))  # the first number is x sample
            train_y.append(float(line[1]))  # the second number is y sample
            line = f.readline().strip()  # read the next line
    ''' Read test data similarly '''
    with open(test_data_file, "r") as f:
        line = f.readline().strip()
        while line:
            line = line.split(' ')
            test_x.append(float(line[0]))
            test_y.append(float(line[1]))
            line = f.readline().strip()
    train_x = np.array(train_x, dtype=float)
    train_y = np.array(train_y, dtype=float)
    test_x = np.array(test_x, dtype=float)
    test_y = np.array(test_y, dtype=float)
    return train_x, train_y, test_x, test_y


def generate_data(my_model,
                  x_train,
                  y_train,
                  x_test,
                  y_test,
                  feature: str,
                  degree: int,
                  start=-20,
                  end=10,
                  steps=500):
    """
    give the relation of RMS and ln_alpha
    """
    ln_alpha_array = np.linspace(start, end, steps)
    RMS_train = []
    RMS_test = []
    params = []
    for ln_alpha in ln_alpha_array:
        my_model.add_mode("ridge", alpha=np.exp(ln_alpha))
        my_model.fit(x_train, y_train, feature, degree)  # train your model
        y_predict_train = my_model.predict(x_train)  # result on your train set
        RMS_train.append(
            my_model.evaluation(y_predict_train, y_train, metric="RMSE"))
        params.append([my_model.model.intercept_, *my_model.model.coef_])
        y_predict_test = my_model.predict(x_test)
        RMS_test.append(
            my_model.evaluation(y_predict_test, y_test, metric="RMSE"))
    return np.array(ln_alpha_array), np.array(RMS_train), \
        np.array(params), np.array(RMS_test)


def main():
    train_data_file = "%s/data/train.dat" % os.path.pardir
    test_data_file = "./test.dat"
    train_x, train_y, test_x, test_y = parse_dataset(
        train_data_file, test_data_file)  # read data from files and parse them

    my_model = MLChemLab1()  # instantiation of the custom class
    # generating train data under polynomial feature
    ln_alpha_array, RMS_train, params, RMS_test = generate_data(my_model,
                                                                train_x,
                                                                train_y,
                                                                test_x,
                                                                test_y,
                                                                "poly",
                                                                4,
                                                                start=-10,
                                                                end=20)
    # plot RMS with poly feature
    plt.plot(ln_alpha_array, RMS_train, label="RMS_train")
    plt.plot(ln_alpha_array, RMS_test, label="RMS_test")
    plt.xlabel("$\\ln(\\lambda)$")
    plt.ylabel("RMSE")
    plt.legend()
    plt.title(
        "RMSE of different $\\lambda$ with polynomial features(degree=4)")
    plt.savefig("RMSE_polynomials.png", dpi=800)
    plt.close()
    # plot coefficients
    plt.plot(ln_alpha_array, params.T[0], label="$C_0$")
    plt.plot(ln_alpha_array, params.T[1], label="$C_1$")
    plt.plot(ln_alpha_array, params.T[2], label="$C_2$")
    plt.plot(ln_alpha_array, params.T[3], label="$C_3$")
    plt.plot(ln_alpha_array, params.T[4], label="$C_4$")
    plt.xlabel("$\\ln(\\lambda)$")
    plt.ylabel("params")
    plt.legend()
    plt.title(
        "parameters of different $\\lambda$ with polynomial features(degree=4)"
    )
    plt.savefig("Params_polynomials.png", dpi=800)
    plt.close()
    # generating train data under poly-cos feature
    ln_alpha_array, RMS_train, params, RMS_test = generate_data(my_model,
                                                                train_x,
                                                                train_y,
                                                                test_x,
                                                                test_y,
                                                                "poly-cos",
                                                                4,
                                                                start=-10,
                                                                end=15)
    # plot RMS with poly-cos feature
    plt.plot(ln_alpha_array, RMS_train, label="RMS_train")
    plt.plot(ln_alpha_array, RMS_test, label="RMS_test")
    plt.xlabel("$\\ln(\\lambda)$")
    plt.ylabel("RMSE")
    plt.legend()
    plt.title("RMSE of different $\\lambda$ with poly-cos features(degree=4)")
    plt.savefig("RMSE_poly-cos.png", dpi=800)
    plt.close()
    # plot coefficients
    plt.plot(ln_alpha_array, params.T[0], label="$C_0$")
    plt.plot(ln_alpha_array, params.T[1], label="$C_1$")
    plt.plot(ln_alpha_array, params.T[2], label="$C_2$")
    plt.plot(ln_alpha_array, params.T[3], label="$C_3$")
    plt.plot(ln_alpha_array, params.T[4], label="$C_4$")
    plt.xlabel("$\\ln(\\lambda)$")
    plt.ylabel("params")
    plt.legend()
    plt.title(
        "parameters of different $\\lambda$ with poly-cos features(degree=4)")
    plt.savefig("Params_poly-cos.png", dpi=800)
    plt.close()

    # generating train data under poly-cos - x feature
    ln_alpha_array, RMS_train, params, RMS_test = generate_data(
        my_model,
        train_x,
        train_y - train_x,
        test_x,
        test_y - test_x,
        "poly-cos",
        4,
        start=-15,
        end=15)
    # plot RMS with poly-cos-x feature
    plt.plot(ln_alpha_array, RMS_train, label="RMS_train")
    plt.plot(ln_alpha_array, RMS_test, label="RMS_test")
    plt.xlabel("$\\ln(\\lambda)$")
    plt.ylabel("RMSE")
    plt.legend()
    plt.title("RMSE of different $\\lambda$ with poly-cos-x features(degree=4)")
    plt.savefig("RMSE_poly-cos-x.png", dpi=800)
    plt.close()
    # plot coefficients
    plt.plot(ln_alpha_array, params.T[0], label="$C_0$")
    plt.plot(ln_alpha_array, params.T[1], label="$C_1$")
    plt.plot(ln_alpha_array, params.T[2], label="$C_2$")
    plt.plot(ln_alpha_array, params.T[3], label="$C_3$")
    plt.plot(ln_alpha_array, params.T[4], label="$C_4$")
    plt.xlabel("$\\ln(\\lambda)$")
    plt.ylabel("params")
    plt.legend()
    plt.title(
        "parameters of different $\\lambda$ with poly-cos-x features(degree=4)")
    plt.savefig("Params_poly-cos-x.png", dpi=800)
    plt.close()


if __name__ == "__main__":
    main()

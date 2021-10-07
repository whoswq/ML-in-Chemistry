# coding=utf-8
"""
Author:         Chongbin Wang
Created Date:   2021/10/7
Last Modified:  2021/10/
"""

from warnings import warn
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt


class My_Linear_model:
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
            return ("model: None\n please use '.add_model()' to choose " /
                    "a type from [lasso, ridge, naive]")
        elif hasattr(self, "__degree"):


    def add_model(self, model: str, **kwargs):
        """
        add model before fitting and prediction

        model: str linear model
        **kwargs:
            alpha: double hyper parameter
        """
        if model == "ridge":
            # Put your Ridge Regression model HERE
            if 'alpha' in kwargs:
                self.model = linear_model.Ridge(alpha=kwargs['alpha'])
            else:
                warn("Hyperparameter alpha for Ridge Regression not found!" /
                     "Using default alpha=0.5")
                self.model = linear_model.Ridge(alpha=0.5)

        elif model == "lasso":
            # Put your Lasso Regression model HERE
            self.model = None
            warn("Lasso has not been implemented yet.")

        elif model == "naive":
            # Put your Linear Regression model HERE
            self.model = linear_model.LinearRegression()
        else:
            # Catch incorrect keywords
            raise NotImplementedError("Got incorrect model keyword " + model)

    def fit(self, x, y, f_mode: str, degree=None):
        """Feature input x using given mode and fit model with featurized X and y

        Args:
            x: Input X data.
            y: Input y data.
            f_mode: Keyword to choose feature methods.
            degree: the degree of polynomials

        Returns:
            Trained model using given x and y, or None if no model is specified
        """

        if (self.model is None):
            warn("No model to fit. Nothing Returned.")
            return None
        self.featurization_mode = f_mode
        self.__degree = degree
        featurized_x = self.__featurization(x, degree=degree)
        self.model.fit(featurized_x, y)
        return self.model

    def __featurization(self, x, **kwargs):
        """Feature input X data using preset mode

        private function, there is no need to call this from outside
        """
        if self.featurization_mode == "poly":
            # Put your polynomial featurization code HERE
            if 'degree' in kwargs and kwargs['degree'] is not None:
                degree = kwargs['degree']
            else:
                warn("Using default degree=4 for polynomial featurization")
                degree = 4  # default quartic polynomial
            x = np.power(x.reshape(-1, 1), np.linspace(
                1, degree,
                degree))  # get polynomial features [[X], [X^2], [X^3], ... ]
            return x

        elif self.featurization_mode == "poly-cos":
            # Put your polynomial cosine featurization code HERE
            raise NotImplementedError("poly-cos has not been implemented yet.")
            return None
        elif self.featurization_mode == "identical":
            # Do nothing, returns raw X data
            return x

    def predict(self, x):
        """Predict based on fitted model and given X data"""
        x = self.featurization(x,
                               featurization_mode=self.featurization_mode,
                               degree=self.degree)
        y_predict = self.model.predict(x)
        return y_predict

    def evaluation(self, y_predict, y_label, metric: str):
            """Eavluate training results based on predicted y and true y"""
            if metric == "RMSE":
                return np.sqrt(
                    np.sum(
                        np.power(y_predict.reshape(-1) - y_label.reshape(-1), 2.0))
                    / y_predict.size)
            else:
                raise NotImplementedError("Got incorrect metric keyword "
                                          + metric)


def parse_dataset(train_data_file, test_data_file):
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    with open(train_data_file, "r") as f:  # open the train data file
        line = f.readline().strip(
        )  # read one line from the file and drop blank characters at the end
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


def main():
    train_data_file = sys.argv[
        1]  # the first argument is the path to training data file
    test_data_file = sys.argv[2]  # and the second argument for test data file
    train_x, train_y, test_x, test_y = parse_dataset(
        train_data_file, test_data_file)  # read data from files and parse them
    my_model = MLChemLab1()  # instantiation of the custom class
    my_model.add_model("ridge", alpha=1)  # add a model to it
    my_model.fit(train_x, train_y, featurization_mode="poly",
                 degree=9)  # fit the model with training data

    y_predict = my_model.predict(
        test_x)  # make prediction with the trained model
    Loss = my_model.evaluation(
        y_predict, test_y,
        metric="RMSE")  # model evaluation with the test dataset
    ''' Plot '''
    plt.scatter(train_x, train_y, color="green", alpha=0.5)
    plt.scatter(test_x, test_y, color="red", s=0.1)
    plt.plot(test_x, y_predict, color="red")
    plt.ylim(-20, 20)
    plt.savefig("chart.png")
    print(Loss)
    print(my_model.model.coef_)


if __name__ == "__main__":
    main()
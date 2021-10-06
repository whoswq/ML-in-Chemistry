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


class My_Regression:
    """
    Lab1 Linear Regression

    Properties:
        model: linear model to fit [lasso, ridge, naive]
        featurization_mode: 
    """
    def __init__(self):
        """
        initialize class with empty model and default featurization mode
        to identical
        """
        self.model = None
        self.featurization_mode = "identical"

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
            degree: 

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
        """Feature input X data using preset mode"""
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
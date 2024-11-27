# Import dependencies
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import numpy as np
from numpy.typing import ArrayLike
from typing import List, Optional

class LogisticRegressionModel:
    def __init__(self, 
                 solver: str,
                 with_cv: bool = False,
                 Cs: Optional[List[float]] = None,
                 cv: Optional[int] = None,
                 scoring: Optional[str] = None,
                 max_iter: Optional[int] = None):
        """
        Initialises a logistic regression model object.

        Arg(s):
        - solver (str): The type of solver to use in the logistic regression model.
        - with_cv (boolean): If True, creates a logistic regression model with cross-validation. Default is False.

        - Cs (Optional[List[float]]): List of C values (regularisation strengths).
        - cv (Optional[int]): The number of cross-validation folds.
        - scoring (Optional[str]): The metric to optimise.
        - max_iter (Optional[int]): The maximum number of iterations for convergence.
        """
        self.solver = solver
        self.with_cv = with_cv
        if self.with_cv:
            self.model = LogisticRegressionCV(
                Cs = Cs,
                cv = cv,
                scoring = scoring,
                max_iter = max_iter
            )
        else:
            self.model = LogisticRegression(solver = solver)


    def preprocess(self, data: List[ArrayLike], labels: List[ArrayLike]) -> tuple[List[ArrayLike], List[ArrayLike]]: 
        """
        Prepares data for logistic regression model 

        Arg(s):
        - data (List[ArrayLike]): The data to be preprocessed
        - labels (List[ArrayLike]): The labels of the data to be preprocessed

        Returns:
        - tuple[List[ArrayLike], List[ArrayLike]]: The preprocessed train and test data and labels
        """
        for i in range(len(data)):
            # Reshape data from 3D to 2D numpy arrays
            data[i] = data[i].reshape(data[i].shape[0], -1)

            # Normalise pixel values to (0, 1)
            data[i] = data[i] / 255.0

        for i in range(len(labels)):
            # Flatten labels
            labels[i] = np.ravel(labels[i])
            
        return data, labels
    

    def predict(self, x_train: ArrayLike, y_train: ArrayLike, x_test: ArrayLike) -> ArrayLike:
        """
        Trains the logistic model and makes classification predictions from the test data

        Arg(s):
        - x_train (ArrayLike): The data with which the model is trained
        - y_train (ArrayLike): The labels of the data with which the model is trained
        - x_test (ArrayLike): The data with which to make a classification prediction after fitting the model

        Returns:
        - ArrayLike: Array of predictions made from test data
        """
        # Train and fit the model
        self.model.fit(x_train, y_train)

        # Predict new values
        y_pred = self.model.predict(x_test)
        
        return y_pred


    def evaluate(self, y_true: ArrayLike, y_pred: ArrayLike):
        """
        Evaluates model accuracy

        Arg(s):
        - y_true (ArrayLike): The test data to be compared
        - y_pred (ArrayLike): The predicted data to be compared
        """
        print(f"Accuracy score: {accuracy_score(y_true, y_pred): .3f}")
        print(f"AUC-ROC Score: {roc_auc_score(y_true, y_pred): .3f}")
        print(classification_report(y_true, y_pred),"\n")   

        if self.with_cv:
            # Evaluating logistic regression with cv
            print(f"Best regularisation value (C): {self.model.C_[0]}\n")


class KNNModel:
    def __init__(self, neighbours: int):
        self.model = KNeighborsClassifier(n_neighbors=neighbours)
    
    def preprocess(self, data: List[ArrayLike], labels: List[ArrayLike]) -> tuple[List[ArrayLike], List[ArrayLike]]: 
        """
        Prepares data for logistic regression model 

        Arg(s):
        - data (List[ArrayLike]): The data to be preprocessed
        - labels (List[ArrayLike]): The labels of the data to be preprocessed

        Returns:
        - tuple[List[ArrayLike], List[ArrayLike]]: The preprocessed train and test data and labels
        """
        for i in range(len(data)):
            # Reshape data from 3D to 2D numpy arrays
            data[i] = data[i].reshape(data[i].shape[0], -1)

            # Normalise pixel values to (0, 1)
            data[i] = data[i] / 255.0

        for i in range(len(labels)):
            # Flatten labels
            labels[i] = np.ravel(labels[i])
            
        return data, labels
    

    def predict(self, x_train: ArrayLike, y_train: ArrayLike, x_test: ArrayLike) -> ArrayLike:
        """
        Trains the logistic model and makes classification predictions from the test data

        Arg(s):
        - x_train (ArrayLike): The data with which the model is trained
        - y_train (ArrayLike): The labels of the data with which the model is trained
        - x_test (ArrayLike): The data with which to make a classification prediction after fitting the model

        Returns:
        - ArrayLike: Array of predictions made from test data
        """
        # Train and fit the model
        self.model.fit(x_train, y_train)

        # Predict new values
        y_pred = self.model.predict(x_test)
        
        return y_pred
    
    def evaluate(self, y_true: ArrayLike, y_pred: ArrayLike):
        """
        Evaluates model accuracy

        Arg(s):
        - y_true (ArrayLike): The test data to be compared
        - y_pred (ArrayLike): The predicted data to be compared
        """
        return roc_auc_score(y_true, y_pred)

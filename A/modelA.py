# Import dependencies
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from numpy.typing import ArrayLike

class LogisticRegressionModel:
    def __init__(self, solver: str):
        """
        Initialises a logistic regression model object

        Arg:
        - solver (str): The type of solver to use in the logistic regression model
        """
        self.solver = solver
        self.model = LogisticRegression(solver = solver)


    def preprocess(self, train_data: ArrayLike, train_labels: ArrayLike, test_data: ArrayLike, test_labels: ArrayLike) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        """
        Prepares data for logistic regression model 

        Arg(s):
        - train_data (ArrayLike): The training data to be preprocessed
        - train_labels (ArrayLike): The labels of the training data to be preprocessed
        - test_data (ArrayLike): The test data to be preprocessed
        - test_labels (ArrayLike): The labels of the test data to be preprocessed

        Returns:
        - tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]: The preprocessed train and test data and labels
        """
        # Reshape data from 3D to 2D numpy arrays
        train_data = train_data.reshape(train_data.shape[0], -1)
        test_data = test_data.reshape(test_data.shape[0], -1 )

        # Normalize pixel values to (0, 1)
        train_data = train_data / 255.0
        test_data = test_data / 255.0

        # Flatten labels
        train_labels = np.ravel(train_labels)
        test_labels = np.ravel(test_labels)

        return train_data, train_labels, test_data, test_labels
    

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


    def evaluate(self, y_test: ArrayLike, y_pred: ArrayLike):
        """
        Evaluates model accuracy

        Arg(s):
        - y_test (ArrayLike): The test data to be compared
        - y_pred (ArrayLike): The predicted data to be compared
        """
        print(f"Accuracy on test set: {accuracy_score(y_test, y_pred)}")
        print(classification_report(y_test, y_pred))

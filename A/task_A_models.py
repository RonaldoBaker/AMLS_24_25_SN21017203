# Import dependencies
import torch
import torch.nn as nn

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.svm import SVC

import numpy as np
from numpy.typing import ArrayLike
from typing import List, Optional
import matplotlib.pyplot as plt


class LogisticRegressionModel:
    def __init__(
        self,
        solver: str,
        with_cv: bool = False,
        Cs: Optional[List[float]] = None,
        cv: Optional[int] = None,
        scoring: Optional[str] = None,
        max_iter: Optional[int] = None,
    ):
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
                Cs=Cs, cv=cv, scoring=scoring, max_iter=max_iter
            )
        else:
            self.model = LogisticRegression(solver=solver)

    # def preprocess(
    #     self, data: List[ArrayLike], labels: List[ArrayLike]
    # ) -> tuple[List[ArrayLike], List[ArrayLike]]:
    #     """
    #     Prepares data for logistic regression model

    #     Arg(s):
    #     - data (List[ArrayLike]): The data to be preprocessed
    #     - labels (List[ArrayLike]): The labels of the data to be preprocessed

    #     Returns:
    #     - tuple[List[ArrayLike], List[ArrayLike]]: The preprocessed train and test data and labels
    #     """
    #     for i in range(len(data)):
    #         # Reshape data from 3D to 2D numpy arrays
    #         data[i] = data[i].reshape(data[i].shape[0], -1)

    #         # Normalise pixel values to (0, 1)
    #         data[i] = data[i] / 255.0

    #     for i in range(len(labels)):
    #         # Flatten labels
    #         labels[i] = np.ravel(labels[i])

    #     return data, labels

    def predict(
        self, x_train: ArrayLike, y_train: ArrayLike, x_test: ArrayLike
    ) -> ArrayLike:
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
        print(f"AUC-ROC Score: {roc_auc_score(y_true, y_pred): .3f}\n")
        if self.with_cv:
            # Evaluating logistic regression with cv
            print(f"Best regularisation value (C): {self.model.C_[0]}\n")

    def report(self, y_true: ArrayLike, y_pred: ArrayLike):
        print(classification_report(y_true, y_pred))


class KNNModel:
    def __init__(self, neighbours: int):
        self.model = KNeighborsClassifier(n_neighbors=neighbours)

    # def preprocess(
    #     self, data: List[ArrayLike], labels: List[ArrayLike]
    # ) -> tuple[List[ArrayLike], List[ArrayLike]]:
    #     """
    #     Prepares data for logistic regression model

    #     Arg(s):
    #     - data (List[ArrayLike]): The data to be preprocessed
    #     - labels (List[ArrayLike]): The labels of the data to be preprocessed

    #     Returns:
    #     - tuple[List[ArrayLike], List[ArrayLike]]: The preprocessed train and test data and labels
    #     """
    #     for i in range(len(data)):
    #         # Reshape data from 3D to 2D numpy arrays
    #         data[i] = data[i].reshape(data[i].shape[0], -1)

    #         # Normalise pixel values to (0, 1)
    #         data[i] = data[i] / 255.0

    #     for i in range(len(labels)):
    #         # Flatten labels
    #         labels[i] = np.ravel(labels[i])

    #     return data, labels

    def predict(
        self, x_train: ArrayLike, y_train: ArrayLike, x_test: ArrayLike
    ) -> ArrayLike:
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


class EarlyStopping:
    def __init__(self, patience: int = 5, delta: int = 0):
        self.patience = patience # How many epochs to wait for improvement
        self.delta = delta # Minimum change in monitored quantity to qualify as improvement
        self.counter = 0 # Counter for patience
        self.best_loss= None # Best score so far
        self.early_stop = False # Whether to stop training

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss # Set best less to the first loss
        elif val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True  # Stop training

class CNNModel(nn.Module):
    def __init__(self):
        """
        Defines the CNN model architecture
        """
        super().__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=2, stride=1)  # First Conv layer
        self.conv2 = nn.Conv2d(3, 16, kernel_size=2, stride=1)  # Second Conv layer
        self.conv3 = nn.Conv2d(16, 32, kernel_size=2, stride=1)  # Third Conv layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Max Pooling
        self.relu = nn.ReLU()  # Activation function
        self.fc1 = nn.Linear(32 * 5 * 5, 512)  # Fully connected layer
        self.fc2 = nn.Linear(512, 256)  # Fully connected layer
        self.fc3 = nn.Linear(256, 128)  # Fully connected layer
        self.fc4 = nn.Linear(128, 1)  # Single output for binary classification
        self.dropout = nn.Dropout(0.4)  # 30% dropout rate
        self.sigmoid = nn.Sigmoid()  # Activation function

    def forward(self, x: ArrayLike) -> ArrayLike:
        """
        Defines how the input goes through the forward pass

        Arg:
        - x (ArrayLike): The data to pass through the neural network

        Returns:
        ArrayLike: The output of the CNN model
        """
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = x.view(x.shape[0], -1)  # Flatten for the fully connected layer
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc4(x))  # Apply sigmoid function for binary classification
        return x


class CNNModelTrainer:
    def __init__(
        self,
        train_data: ArrayLike,
        test_data: ArrayLike,
        val_data: ArrayLike,
        cnn_model: CNNModel,
        epochs: int,
        loss_func: torch.nn,
        optimiser: torch.optim,
    ):
        self.train_data = train_data
        self.test_data = test_data
        self.val_data = val_data
        self.cnn = cnn_model
        self.epochs = epochs
        self.loss_func = loss_func
        self.optimiser = optimiser
        # Empty lists to track loss and accuracy
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []

    def train(self, patience: int = 5):
        # TODO: Comment and add docstring

        # Create instance of EarlyStopping class
        early_stopping = EarlyStopping(patience=patience)

        for epoch in range(self.epochs):
            running_train_loss = 0.0
            train_batch_count = 0
            running_val_loss = 0.0
            running_val_accuracy = 0.0
            val_batch_count = 0

            # Train in batches
            self.cnn.train()
            for _, (images, labels) in enumerate(self.train_data):
                self.optimiser.zero_grad()
                outputs = self.cnn(images)
                loss = self.loss_func(outputs, labels)
                loss.backward()
                self.optimiser.step()
                running_train_loss += loss.item()
                train_batch_count += 1

            # Evaluation
            with torch.no_grad():
                self.cnn.eval()
                for images, labels in self.val_data:
                    self.optimiser.zero_grad()
                    outputs = self.cnn(images)
                    predicted = (outputs > 0.5).float()
                    accuracy = roc_auc_score(labels.cpu(), predicted.cpu())
                    loss = self.loss_func(outputs, labels)
                    running_val_accuracy += accuracy
                    running_val_loss += loss.item()
                    val_batch_count += 1

            # Early stopping
            early_stopping(running_val_loss / val_batch_count)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch: {epoch}")
                break

            self.train_losses.append(running_train_loss / train_batch_count)
            self.val_losses.append(running_val_loss / val_batch_count)
            self.val_accuracies.append(running_val_accuracy / val_batch_count)

            if epoch % 100 == 0:
                print(f"Epoch: {epoch} | Train Loss: {self.train_losses[-1]: .3f} | Val Loss: {self.val_losses[-1]: .3f} | Val Accuracy: {self.val_accuracies[-1]: .3f}")
        print("\n")


    def evaluate(self):
        # TODO: Comment and add docstring
        all_predictions = []
        all_labels = []

        running_accuracy = 0.0
        batch_count = 0
        with torch.no_grad():
            for image, labels in self.test_data:
                outputs = self.cnn(image)
                predicted_labels = (outputs > 0.5).int()
                accuracy = roc_auc_score(labels.cpu(), predicted_labels.cpu())
                all_predictions.extend(predicted_labels.cpu())
                all_labels.extend(labels.cpu())
                running_accuracy += accuracy
                batch_count += 1

        avg_accuracy = running_accuracy / batch_count
        print(f"Accuracy on test data: {avg_accuracy * 100: .2f}%\n")
        print("Classification Report (CNN)")
        print(classification_report(all_labels, all_predictions, zero_division=0))


    def plot_training_curve(self):
        # Plot the training curve
        plt.plot(range(1, len(self.train_losses)+1, 1), self.train_losses, label="Training Loss")
        plt.plot(range(1, len(self.val_losses)+1, 1), self.val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
        plt.title("CNN Training Curve")
        plt.savefig("figures/CNN_Training_Curve.png")


class SVMModel:
    def __init__(self):
        self.model = SVC()

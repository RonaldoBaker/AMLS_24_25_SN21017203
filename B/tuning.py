"""
Program that tunes the hyperparameters of a specified model
"""
import sys
import os

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from acquisitionB import load_bloodmnist_data
from A.preprocessingA import preprocess_for_traditional

TUNE_KNN = True
TUNE_SVM = False

# Load BloodMNIST data
data = load_bloodmnist_data(datapath="Datasets/bloodmnist.npz")

# Separate data
train_data = data["train_data"]
train_labels = data["train_labels"]
test_data = data["test_data"]
test_labels = data["test_labels"]
val_data = data["val_data"]
val_labels = data["val_labels"]

data, labels = preprocess_for_traditional(data = [train_data, test_data], labels=[train_labels, test_labels])
X_train, X_test = data[0], data[1]
y_train, y_test = labels[0], labels[1]


if TUNE_KNN:
    print("Tuning KNN hyperparameters\n")
    knn = KNeighborsClassifier()

    # Define hyperparameter grid
    parameter_grid = {
        "n_neighbors": np.arange(1, 31),
        "weights": ["uniform", "distance"],
        "algorithm": ["ball_tree", "kd_tree", "brute"],
    }

    # Perform grid search
    grid_search = RandomizedSearchCV(estimator=knn, param_distributions=parameter_grid, n_jobs=-1, scoring="accuracy")
    grid_search.fit(X_train, y_train.ravel())

    # Get the best parameters and corresponding accuracy score
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best accuracy score: {grid_search.best_score_}")


if TUNE_SVM:
    print("Tuning SVM hyperparameters\n")
    svm = SVC(class_weight="balanced")

    # Define hyperparameter grid
    parameter_grid = {
        "C": [0.1, 1, 10],
        "kernel": ["rbf", "poly"],
        "gamma": ["scale", "auto"],
        "degree": [2, 3, 4],
        "decision_function_shape": ["ovo", "ovr"],
    }

    # Perform grid search
    grid_search = RandomizedSearchCV(estimator=svm, param_distributions=parameter_grid, n_jobs=-1, scoring="accuracy", random_state=7)
    grid_search.fit(X_train, y_train.ravel())

    # Get the best parameters and corresponding accuracy score
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best accuracy score: {grid_search.best_score_}")
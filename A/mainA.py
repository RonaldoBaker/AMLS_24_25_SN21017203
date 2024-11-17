# Import dependencies
from A.acquisitionA import load_breastmnist_data, analyse
from A.modelA import LogisticRegressionModel
import numpy as np


def taskA():
    """
    Executes task A, including data loading and model training/evaluation
    """
    # Define constants
    DATAPATH = "Datasets/breastmnist.npz"
    SOLVER = "lbfgs"

    # Load BreastMNIST data
    data = load_breastmnist_data(datapath=DATAPATH)

    # Analyse data
    analyse(data)

    # Separate data
    train_data = data["train_data"]
    train_labels = data["train_labels"]
    test_data = data["test_data"]
    test_labels = data["test_labels"]
    val_data = data["val_data"]
    val_labels = data["val_labels"]

    # Instantiate model
    logreg = LogisticRegressionModel(solver = SOLVER)

    # Preprocess data
    processed_data, processed_labels = logreg.preprocess(data = [train_data, val_data, test_data], labels = [train_labels, val_labels, test_labels])
    train_data, val_data, test_data = processed_data[0], processed_data[1], processed_data[2]
    train_labels, val_labels, test_labels = processed_labels[0], processed_labels[1], processed_labels[2]

    # Make prediction with validation data
    predict_labels_val = logreg.predict(train_data, train_labels, val_data)

    # Evaluation prediction
    print("Evaluation on validation set")
    logreg.evaluate(val_labels, predict_labels_val)

    # Make classification prediction on test data
    predict_labels = logreg.predict(train_data, train_labels, test_data)

    # Evaluate prediction
    print("Evaluation on test set")
    logreg.evaluate(test_labels, predict_labels)

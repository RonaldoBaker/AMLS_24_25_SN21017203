# Import dependencies
from A.acquisitionA import load_breastmnist_data
from A.modelA import LogisticRegressionModel


def taskA():
    """
    Executes task A, including data loading and model training/evaluation
    """
    # Define constants
    DATAPATH = "Datasets/breastmnist.npz"
    SOLVER = "lbfgs"

    # Load BreastMNIST data
    data = load_breastmnist_data(datapath=DATAPATH)

    # Separate data
    train_data = data["train_data"]
    train_labels = data["train_labels"]
    test_data = data["test_data"]
    test_labels = data["test_labels"]

    # Instantiate model
    logreg = LogisticRegressionModel(solver = SOLVER)

    # Preprocess data
    train_data, train_labels, test_data, test_labels = logreg.preprocess(train_data, train_labels, test_data, test_labels)

    # Make classification prediction on test data
    predict_labels = logreg.predict(train_data, train_labels, test_data)

    # Evaluate prediction
    logreg.evaluate(test_labels, predict_labels)
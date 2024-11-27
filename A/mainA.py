# Import dependencies
from A.acquisitionA import load_breastmnist_data, analyse
from A.task_A_models import LogisticRegressionModel, KNNModel
import numpy as np
import matplotlib.pyplot as plt


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

    # Instantiate model logistic regression without cross-validation
    logreg = LogisticRegressionModel(solver = SOLVER)

    # Preprocess data
    processed_data, processed_labels = logreg.preprocess(data = [train_data, val_data, test_data], labels = [train_labels, val_labels, test_labels])
    train_data, val_data, test_data = processed_data[0], processed_data[1], processed_data[2]
    train_labels, val_labels, test_labels = processed_labels[0], processed_labels[1], processed_labels[2]

    # Make prediction with validation data
    y_val_pred= logreg.predict(train_data, train_labels, val_data)

    # Evaluation prediction
    print("Evaluation on validation set")
    logreg.evaluate(val_labels, y_val_pred)

    # Make classification prediction on test data
    y_test_pred = logreg.predict(train_data, train_labels, test_data)

    # Evaluate prediction
    print("Evaluation on test set")
    logreg.evaluate(test_labels, y_test_pred)

    # ------------------------------------------------------------------- #

    # Instantiate logistic regression model with cross-validation
    logreg_cv = LogisticRegressionModel(solver = SOLVER,
                                        with_cv = True,
                                        Cs = [0.001, 0.01, 0.1, 1, 10, 100],
                                        cv = 5,
                                        scoring = "roc_auc",
                                        max_iter = 1000)
    
    y_val_pred_cv = logreg_cv.predict(train_data, train_labels, val_data)
    
    # TODO:This value needs to be passed directly to the roc_auc_score function, not the value above
    y_val_auc_pred = logreg_cv.model.predict_proba(val_data)[:, 1]

    logreg_cv.evaluate(val_labels, y_val_pred_cv)

    # ------------------------------------------------------------------- #

    # KNN model - Finding the optimum value of K (the number of nearest neighbours)
    accuracies = []


    NEIGHBOURS = 30
    for k in range(1, NEIGHBOURS+1):
        knn_model = KNNModel(neighbours=k)
        y_pred = knn_model.predict(train_data, train_labels, test_data)
        accuracies.append(knn_model.evaluate(test_labels, y_pred))

    # Plot number of nearest neighbours vs AUC-ROC accuracy
    plt.plot(range(1, NEIGHBOURS+1), accuracies, marker = 'o')
    plt.grid()
    plt.title("Accuracy vs K Value")
    plt.xlabel("No. of nearest neighbours")
    plt.ylabel("AUC-ROC Accuracy Score")
    plt.show()
 
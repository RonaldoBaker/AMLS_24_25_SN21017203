# Import dependencies
from A.acquisitionA import load_breastmnist_data, analyse
from A.task_A_models import LogisticRegressionModel, KNNModel, CNNModel, CNNModelTrainer
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim


def taskA():
    """
    Executes task A, including data loading and model training/evaluation
    """
    # Define constants
    DATAPATH = "Datasets/breastmnist.npz"
    SOLVER = "lbfgs"
    RUN_KNN = False
    RUN_LOGREG = False
    RUN_CNN = True

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
    if RUN_LOGREG:
        print("LOGISTIC REGRESSION\n")
        logreg = LogisticRegressionModel(solver = SOLVER)

        # Preprocess data
        processed_data, processed_labels = logreg.preprocess(data = [train_data, val_data, test_data], labels = [train_labels, val_labels, test_labels])
        processed_train_data, processed_val_data, processed_test_data = processed_data[0], processed_data[1], processed_data[2]
        processed_train_labels, processed_val_labels, processed_test_labels = processed_labels[0], processed_labels[1], processed_labels[2]

        # Make prediction with validation data
        y_val_pred= logreg.predict(processed_train_data, processed_train_labels, processed_val_data)

        # Evaluation prediction
        print("Evaluation on validation set")
        logreg.evaluate(processed_val_labels, y_val_pred)

        # Make classification prediction on test data
        y_test_pred = logreg.predict(processed_train_data, processed_train_labels, processed_test_data)

        # Evaluate prediction
        print("Evaluation on test set")
        logreg.evaluate(processed_test_labels, y_test_pred)
        print("Classification Report (Logistic Regression)")
        logreg.report(processed_test_labels, y_test_pred)

        # ------------------------------------------------------------------- #

        # Instantiate logistic regression model with cross-validation
        print("LOGISTIC REGRESSION WITH CROSS-VALIDATION\n")
        logreg_cv = LogisticRegressionModel(solver = SOLVER,
                                            with_cv = True,
                                            Cs = [0.001, 0.01, 0.1, 1, 10, 100],
                                            cv = 5,
                                            scoring = "roc_auc",
                                            max_iter = 1000)
        
        y_val_pred_cv = logreg_cv.predict(processed_train_data, processed_train_labels, processed_val_data)
        
        # TODO:This value needs to be passed directly to the roc_auc_score function, not the value above
        # y_val_auc_pred = logreg_cv.model.predict_proba(processed_val_data)[:, 1]

        print("Evaluation on validation set")
        logreg_cv.evaluate(processed_val_labels, y_val_pred_cv)

        print("Classification Report (Logistic Regression with Cross-Validation)")
        logreg_cv.report(processed_val_labels, y_val_pred_cv)

    # ------------------------------------------------------------------- #

    # KNN model - Finding the optimum value of K (the number of nearest neighbours)
    if RUN_KNN:
        print("K-NEAREST NEIGHBOURS\n")
        accuracies = []
        NEIGHBOURS = 30

        for k in range(1, NEIGHBOURS+1):
            knn_model = KNNModel(neighbours=k)
            y_pred = knn_model.predict(processed_train_data, processed_train_labels, processed_test_data)
            accuracies.append(knn_model.evaluate(processed_test_labels, y_pred))

        # Plot number of nearest neighbours vs AUC-ROC accuracy
        plt.plot(range(1, NEIGHBOURS+1), accuracies, marker = 'o', linestyle = '--', color = 'b')
        plt.grid()
        plt.title("Accuracy vs K Value")
        plt.xlabel("No. of nearest neighbours")
        plt.ylabel("AUC-ROC Score")
        plt.savefig("figures/KNN_Accuracy_vs_K.png")
 
    # ------------------------------------------------------------------- #

    # CNN from scratch
    # Define hyperparameters
    BATCH_SIZE = 64
    EPOCHS = 1000
    LEARNING_RATE = 0.001

    if RUN_CNN:
        if torch.cuda.is_available():
            # Define device if dedicated GPU is available
            DEVICE_NUM = 0
            torch.cuda.set_device(DEVICE_NUM)
            DEVICE = torch.device(f"cuda:{DEVICE_NUM}")
        else:
            DEVICE = torch.device("cpu")

        # Create tensors and add dimension for greyscale image data, and make labels 2D
        train_data_tensor = torch.tensor(train_data, device=DEVICE, dtype=torch.float32).unsqueeze(1)
        train_labels_tensor = torch.tensor(train_labels, device=DEVICE, dtype=torch.float32)

        test_data_tensor = torch.tensor(test_data, device=DEVICE, dtype=torch.float32).unsqueeze(1)
        test_labels_tensor = torch.tensor(test_labels, device=DEVICE, dtype=torch.float32)

        val_data_tensor = torch.tensor(val_data, device=DEVICE, dtype=torch.float32).unsqueeze(1)
        val_labels_tensor = torch.tensor(val_labels, device=DEVICE, dtype=torch.float32)

        # Create DataLoaders 
        train_set = [(train_data_tensor[i], train_labels_tensor[i]) for i in range(len(train_data_tensor))]
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

        test_set = [(test_data_tensor[i], test_labels_tensor[i]) for i in range(len(test_data_tensor))]
        test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

        val_set = [(val_data_tensor[i], val_labels_tensor[i]) for i in range(len(val_data_tensor))]
        val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

        # Instantiate CNN model and move to device being used
        print("CNN\n")
        cnn = CNNModel()
        cnn.to(DEVICE) 

        # Define loss function and optimiser
        loss_func = nn.BCELoss()
        optimiser = optim.Adam(cnn.parameters(), lr=LEARNING_RATE)

        # Train model
        cnn_trainer = CNNModelTrainer(train_loader, test_loader, val_loader, cnn, EPOCHS, loss_func, optimiser)
        cnn_trainer.train()
        cnn_trainer.evaluate()
        cnn_trainer.plot_training_curve()

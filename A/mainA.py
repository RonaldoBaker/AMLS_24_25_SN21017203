# Import dependencies
from A.acquisitionA import load_breastmnist_data, display_info
from A.preprocessingA import preprocess
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
    RUN_KNN = True
    RUN_LOGREG = True
    RUN_CNN = False

    # Load BreastMNIST data
    data = load_breastmnist_data(datapath=DATAPATH)

    # Display data
    display_info(data)

    # Separate data
    train_data = data["train_data"]
    train_labels = data["train_labels"]
    test_data = data["test_data"]
    test_labels = data["test_labels"]
    val_data = data["val_data"]
    val_labels = data["val_labels"]

    # Preprocess data
    data, labels = preprocess(data = [train_data, val_data, test_data], labels = [train_labels, val_labels, test_labels])
    X_train, X_val, X_test = data[0], data[1], data[2]
    y_train, y_val, y_test = labels[0], labels[1], labels[2]

    # Instantiate model logistic regression without cross-validation
    if RUN_LOGREG:
        print("LOGISTIC REGRESSION\n")
        logreg = LogisticRegressionModel(solver = SOLVER)

        # Make classification prediction on test data
        y_test_pred = logreg.predict(X_train, y_train, X_test)

        # Evaluate prediction
        print("Evaluation on test set")
        logreg.evaluate(y_test, y_test_pred)
        print("Classification Report (Logistic Regression)")
        logreg.report(y_test, y_test_pred)

    # ------------------------------------------------------------------- #

    # KNN model - Finding the optimum value of K (the number of nearest neighbours)
    if RUN_KNN:
        print("K-NEAREST NEIGHBOURS\n")
        accuracies = []
        NEIGHBOURS = 30

        for k in range(1, NEIGHBOURS+1):
            knn_model = KNNModel(neighbours=k)
            y_pred = knn_model.predict(X_train, y_train, X_test)
            accuracies.append(knn_model.evaluate(y_test, y_pred))

        # Plot number of nearest neighbours vs AUC-ROC accuracy
        plt.plot(range(1, NEIGHBOURS+1), accuracies, marker = 'o', linestyle = '--', color = 'b')
        plt.grid()
        plt.title("Accuracy vs K Value")
        plt.xlabel("No. of nearest neighbours")
        plt.ylabel("AUC-ROC Score")
        plt.savefig("figures/KNN_Accuracy_vs_K.png")
 
    # ------------------------------------------------------------------- #

    # CNN model
    # Define hyperparameters
    BATCH_SIZE = 64
    EPOCHS = 1000
    LEARNING_RATE = 0.001
    RANDOM_SEED = 7

    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if RUN_CNN:
        if torch.cuda.is_available():
            # Define device if dedicated GPU is available
            DEVICE_NUM = 1
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
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

        test_set = [(test_data_tensor[i], test_labels_tensor[i]) for i in range(len(test_data_tensor))]
        test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

        val_set = [(val_data_tensor[i], val_labels_tensor[i]) for i in range(len(val_data_tensor))]
        val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

        # Instantiate CNN model and move to device being used
        print("CNN\n")
        cnn = CNNModel()
        cnn.to(DEVICE) 

        # Define loss function and optimiser
        loss_func = nn.BCELoss()
        optimiser = optim.Adam(cnn.parameters(), lr=LEARNING_RATE)

        # Train model
        cnn_trainer = CNNModelTrainer(train_loader, test_loader, val_loader, cnn, EPOCHS, loss_func, optimiser)
        cnn_trainer.train(patience=3)
        cnn_trainer.evaluate()
        cnn_trainer.plot_training_curve()

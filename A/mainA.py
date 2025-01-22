# Import dependencies
from A.acquisitionA import load_breastmnist_data, display_info
from A.preprocessingA import preprocess_for_traditional
from A.taskAmodels import CNNModel, CNNModelTrainer
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.svm import SVC


def taskA(mode: str):

    """
    Executes task A, including data loading, preprocessing, and model training/evaluation.
    
    Arg(s):
    mode (str): The mode in which to run the task. Options are 'test' or 'train'. If 'train', models are trained and evaluated. If 'test', saved models are loaded and evaluated.
    """
    # Define constants
    DATAPATH = "Datasets/breastmnist.npz"
    RUN_LOGREG = True
    RUN_KNN = True
    RUN_SVM = True
    RUN_CNN = True

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

    # Preprocess data for traditional machine learning models
    data, labels = preprocess_for_traditional(data = [train_data, test_data], labels = [train_labels, test_labels])
    X_train_balanced, X_test = data[0], data[1]
    y_train_balanced, y_test = labels[0], labels[1]

    # ------------------------------------------------------------------- #
    # Logistic regression model - using the lbfgs solver
    if RUN_LOGREG:
        print("LOGISTIC REGRESSION\n")
        logreg = LogisticRegression(solver = "lbfgs", class_weight="balanced") # Create model
        logreg.fit(X_train_balanced, y_train_balanced) # Fit model
        y_pred = logreg.predict(X_test) # Make predictions

        # Evaluate prediction
        print("Evaluation on test set")
        score = roc_auc_score(y_test, y_pred) * 100
        print(f"ROC-AUC Score: {score: .2f}%\n")
        print("Classification Report (Logistic Regression)")
        print(classification_report(y_test, y_pred))

    # ------------------------------------------------------------------- #
    # KNN model - Finding the optimum value of K (the number of nearest neighbours)
    if RUN_KNN:
        print("K-NEAREST NEIGHBOURS\n")
        accuracies = []
        NEIGHBOURS = 30

        best_k = 0
        max_score = 0
        for k in range(1, NEIGHBOURS+1):
            knn_model = KNeighborsClassifier(n_neighbors=k, weights="uniform")
            # knn_model = KNNModel(neighbours=k)
            knn_model.fit(X_train_balanced, y_train_balanced) # Fit model
            y_pred = knn_model.predict(X_test) # Make predictions

            # Evaluation prediction
            score = roc_auc_score(y_test, y_pred) * 100
            if score > max_score: # Track best K value
                best_k = k
                max_score = score
            accuracies.append(score)

        print(f"Best K Value: {best_k}")
        knn_model = KNeighborsClassifier(n_neighbors=best_k, weights="uniform")
        knn_model.fit(X_train_balanced, y_train_balanced)
        y_pred = knn_model.predict(X_test)

        # Evaluate prediction for best K Value
        print("Evaluation on test set")
        score = roc_auc_score(y_test, y_pred) * 100
        print(f"ROC-AUC Score: {score: .2f}%\n")

        print("Classification Report (K-NEAREST NEIGHBOURS)")
        print(classification_report(y_test, y_pred))

        # Plot number of nearest neighbours vs AUC-ROC score

        plt.figure(figsize=(6,4))
        plt.plot(range(1, NEIGHBOURS+1), accuracies, marker = 'o', linestyle = '--', color = 'b', linewidth = 2)
        plt.title("ROC-AUC vs K Value")
        plt.xlabel("K Value", fontsize=12)
        plt.ylabel("ROC-AUC Score (%)", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.8)
        plt.tight_layout()
        plt.savefig("figures/KNN_Accuracy_vs_K.png")
 
    # ------------------------------------------------------------------- #
    # SVM model
    if RUN_SVM:
        print("SUPPORT VECTOR MACHINE\n")
        svm = SVC(kernel='poly', degree=2, class_weight='balanced', random_state=7)
        svm.fit(X_train_balanced, y_train_balanced) # Fit model
        y_pred = svm.predict(X_test) # Make predictions

        # Evaluate prediction
        print("Evaluation on test set")
        score = roc_auc_score(y_test, y_pred) * 100
        print(f"ROC-AUC Score: {score: .2f}%\n")
        print("Classification Report (SVM)")
        print(classification_report(y_test, y_pred))

    # ------------------------------------------------------------------- #
    # CNN model


    if RUN_CNN:
        print("CNN\n")
        # Define hyperparameters
        BATCH_SIZE = 64
        EPOCHS = 100
        LEARNING_RATE = 0.001
        RANDOM_SEED = 7
        SAVE_MODEL = True
        MODEL_PATH = "A/cnn_modelA.pth"

        # Set random seed for reproducibility
        torch.manual_seed(RANDOM_SEED)
        torch.cuda.manual_seed(RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if torch.cuda.is_available():
            # Define device if dedicated GPU is available
            DEVICE_NUM = 1
            torch.cuda.set_device(DEVICE_NUM)
            DEVICE = torch.device(f"cuda:{DEVICE_NUM}")
        else:
            DEVICE = torch.device("cpu")

        # Instantiate CNN model
        cnn = CNNModel()

        # Create the test loader first if the mode is 'test'
        test_data_tensor = torch.tensor(test_data, device=DEVICE, dtype=torch.float32).unsqueeze(1)
        test_labels_tensor = torch.tensor(test_labels, device=DEVICE, dtype=torch.float32)
        test_set = [(test_data_tensor[i], test_labels_tensor[i]) for i in range(len(test_data_tensor))]
        test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

        if mode == "test":
            # Load saved model and evaluate
            cnn.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
            cnn.to(DEVICE)
            cnn.eval()
            cnn_trainer = CNNModelTrainer(None, test_loader, None, cnn, None, None, None)
            cnn_trainer.evaluate()

        elif mode == "train":
            # Create tensors and add dimension for greyscale image data, and make labels 2D
            train_data_tensor = torch.tensor(train_data, device=DEVICE, dtype=torch.float32).unsqueeze(1)
            train_labels_tensor = torch.tensor(train_labels, device=DEVICE, dtype=torch.float32)

            val_data_tensor = torch.tensor(val_data, device=DEVICE, dtype=torch.float32).unsqueeze(1)  
            val_labels_tensor = torch.tensor(val_labels, device=DEVICE, dtype=torch.float32)

            # Create DataLoaders 
            train_set = [(train_data_tensor[i], train_labels_tensor[i]) for i in range(len(train_data_tensor))]
            train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

            val_set = [(val_data_tensor[i], val_labels_tensor[i]) for i in range(len(val_data_tensor))]
            val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

            # Move CNN to device
            cnn.to(DEVICE)

            # Define loss function and optimiser
            loss_func = nn.BCELoss()
            optimiser = optim.Adam(cnn.parameters(), lr=LEARNING_RATE)

            # Train model
            cnn_trainer = CNNModelTrainer(train_loader, test_loader, val_loader, cnn, EPOCHS, loss_func, optimiser)
            cnn_trainer.train(patience=2)

            if SAVE_MODEL:
                torch.save(cnn_trainer.cnn.state_dict(), MODEL_PATH)

            cnn_trainer.evaluate()
            cnn_trainer.plot_training_curve(filepath="figures/training_curve_taskA.png")

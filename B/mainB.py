import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import RandomizedSearchCV
from B.acquisitionB import load_bloodmnist_data, display_info
from B.preprocessingB import preprocess_for_traditional
from B.taskBmodels import CNNModel, CNNModelTrainer

def taskB():
    """
    Executes task B, including data loading, processing and model training/evaluation
    """
    # Define constants
    RUN_SVM = False
    RUN_CNN = True

    # Load BloodMNIST data
    data = load_bloodmnist_data(datapath="Datasets/bloodmnist.npz")

    # Display data
    display_info(data)

    # Separate data
    train_data = data["train_data"]
    train_labels = data["train_labels"]
    test_data = data["test_data"]
    test_labels = data["test_labels"]
    val_data = data["val_data"]
    val_labels = data["val_labels"]

    data = preprocess_for_traditional(data = [train_data, test_data])
    X_train, X_test = data[0], data[1]
    y_train, y_test = train_labels, test_labels


    if RUN_SVM:
        print("\nSVM\n")

        # Code to perform randomised search for hyperparameter tuning
        # Commented out for the sake of time
        """
        # Create SVM model
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
        
        # Evaluate SVM model
        print("Evaluation on test set")
        best_svm = grid_search.best_estimator_
        y_pred = best_svm.predict(X_test)
        score = accuracy_score(y_test, y_pred) * 100
        print(f"Accuracy Score: {score: .2f}%\n")
        print("Classification Report (SVM)")
        print(classification_report(y_test, y_pred))
        """

        # Redefine SVM model with best parameters
        svm = SVC(kernel="poly", degree=4, gamma="scale", decision_function_shape="ovr", class_weight="balanced")
        svm.fit(X_train, y_train.ravel())

        # Evaluate SVM model
        print("Evaluation on test set")
        y_pred = svm.predict(X_test)
        score = accuracy_score(y_test, y_pred) * 100
        print(f"Accuracy Score: {score: .2f}%\n")
        print("Classification Report (SVM)")
        print(classification_report(y_test, y_pred))


    if RUN_CNN:
        print("CNN\n")
        # CNN model from Task A
        BATCH_SIZE = 128
        EPOCHS = 1000
        LEARNING_RATE = 0.0004
        RANDOM_SEED = 7

        # Set device
        if torch.cuda.is_available():
            DEVICE_NUM = 1
            torch.cuda.set_device(DEVICE_NUM)
            DEVICE = torch.device(f"cuda:{DEVICE_NUM}")
        else:
            DEVICE = torch.device("cpu")

        # Set random seed for reproducibility
        torch.manual_seed(RANDOM_SEED)
        torch.cuda.manual_seed(RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


        # Reshape data to be channel first
        train_data = train_data.transpose(0, 3, 1, 2)
        test_data = test_data.transpose(0, 3, 1, 2)
        val_data = val_data.transpose(0, 3, 1, 2)

        # Create tensors and add dimension for greyscale image data, and make labels 2D
        train_data_tensor = torch.tensor(train_data, device=DEVICE, dtype=torch.float32)
        train_labels_tensor = torch.tensor(train_labels, device=DEVICE, dtype=torch.float32)

        test_data_tensor = torch.tensor(test_data, device=DEVICE, dtype=torch.float32)
        test_labels_tensor = torch.tensor(test_labels, device=DEVICE, dtype=torch.float32)

        val_data_tensor = torch.tensor(val_data, device=DEVICE, dtype=torch.float32)  
        val_labels_tensor = torch.tensor(val_labels, device=DEVICE, dtype=torch.float32)
        
        # Create DataLoaders 
        train_set = [(train_data_tensor[i], train_labels_tensor[i]) for i in range(len(train_data_tensor))]
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

        test_set = [(test_data_tensor[i], test_labels_tensor[i]) for i in range(len(test_data_tensor))]
        test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

        val_set = [(val_data_tensor[i], val_labels_tensor[i]) for i in range(len(val_data_tensor))]
        val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

        # Instantiate CNN model and move to device being used
        cnn = CNNModel()
        cnn.to(DEVICE) 

        # Define loss function and optimiser
        loss_func = nn.CrossEntropyLoss()
        optimiser = optim.Adam(cnn.parameters(), lr=LEARNING_RATE)

        # Train model
        cnn_trainer = CNNModelTrainer(train_loader, test_loader, val_loader, cnn, EPOCHS, loss_func, optimiser)
        cnn_trainer.train(patience=3)
        cnn_trainer.evaluate()
        cnn_trainer.plot_training_curve("figures/training_curve_taskB.png")

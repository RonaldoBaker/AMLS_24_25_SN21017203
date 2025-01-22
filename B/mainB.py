import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import label_binarize
from sklearn.neighbors import KNeighborsClassifier
from B.acquisitionB import load_bloodmnist_data, display_info
from A.preprocessingA import preprocess_for_traditional
from B.taskBmodels import CNNModel, CNNModelTrainer

def taskB(mode: str):
    """
    Executes task B, including data loading, processing and model training/evaluation
    """
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

    data, labels = preprocess_for_traditional(data = [train_data, test_data], labels=[train_labels, test_labels])
    X_train, X_test = data[0], data[1]
    y_train, y_test = labels[0], labels[1]

    # ---------------------------------------------------- #
    # KNN model
    print("\nKNN\n")
    # Define KNN model with best parameters
    knn = KNeighborsClassifier(n_neighbors=4, weights="distance", algorithm="kd_tree")
    knn.fit(X_train, y_train.ravel())

    # Evaluate KNN model
    print("Evaluation on test set")
    y_pred_proba = knn.predict_proba(X_test)

    # Convert test labels to one-hot encoded format
    y_test_one_hot = label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5, 6, 7])

    score = roc_auc_score(y_test_one_hot, y_pred_proba, multi_class="ovr") * 100
    print(f"Accuracy Score: {score: .2f}%\n")
    print("Classification Report (KNN)")
    y_pred = y_pred_proba.argmax(axis=1)
    print(classification_report(y_test, y_pred))

    # ---------------------------------------------------- #
    # SVM model
    print("\nSVM\n")
    # Define SVM model with best parameters
    svm = SVC(C=0.1, kernel="poly", degree=4, gamma="scale", decision_function_shape="ovr", class_weight="balanced", probability=True)
    svm.fit(X_train, y_train.ravel())

    # Evaluate SVM model
    print("Evaluation on test set")
    y_pred_proba = svm.predict_proba(X_test)

    # Convert test labels to one-hot encoded format
    y_test_one_hot = label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5, 6, 7])

    score = roc_auc_score(y_test_one_hot, y_pred_proba) * 100
    print(f"Accuracy Score: {score: .2f}%\n")
    print("Classification Report (SVM)")
    y_pred = y_pred_proba.argmax(axis=1)
    print(classification_report(y_test, y_pred))

    # ---------------------------------------------------- #
    # CNN model
    print("CNN\n")
    # CNN model from Task A
    BATCH_SIZE = 128
    EPOCHS = 100
    LEARNING_RATE = 0.0004
    RANDOM_SEED = 7
    SAVE_MODEL = True
    MODEL_PATH = "B/cnn_modelB.pth"

    # Set random seed for reproducibility
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set device
    if torch.cuda.is_available():
        DEVICE_NUM = 1
        torch.cuda.set_device(DEVICE_NUM)
        DEVICE = torch.device(f"cuda:{DEVICE_NUM}")
    else:
        DEVICE = torch.device("cpu")

    # Instantiate CNN model 
    cnn = CNNModel()

    # Create test loader first if the mode is 'test'
    test_data = test_data.transpose(0, 3, 1, 2)
    test_data_tensor = torch.tensor(test_data, device=DEVICE, dtype=torch.float32)
    test_labels_tensor = torch.tensor(test_labels, device=DEVICE, dtype=torch.float32)
    test_set = [(test_data_tensor[i], test_labels_tensor[i]) for i in range(len(test_data_tensor))]
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    
    if mode == "test":
        # Load saved model and evaluate
        cnn = CNNModel()
        cnn.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
        cnn.to(DEVICE)
        cnn.eval()
        cnn_trainer = CNNModelTrainer(None, test_loader, None, cnn, None, None, None)
        cnn_trainer.evaluate()

    elif mode == "train":
        # Transpose data to be channel first
        train_data = train_data.transpose(0, 3, 1, 2)
        val_data = val_data.transpose(0, 3, 1, 2)

        # Create tensors and add dimension for greyscale image data, and make labels 2D
        train_data_tensor = torch.tensor(train_data, device=DEVICE, dtype=torch.float32)
        train_labels_tensor = torch.tensor(train_labels, device=DEVICE, dtype=torch.float32)

        val_data_tensor = torch.tensor(val_data, device=DEVICE, dtype=torch.float32)
        val_labels_tensor = torch.tensor(val_labels, device=DEVICE, dtype=torch.float32)
        
        # Create DataLoaders 
        train_set = [(train_data_tensor[i], train_labels_tensor[i]) for i in range(len(train_data_tensor))]
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

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

        if SAVE_MODEL:
            torch.save(cnn_trainer.cnn.state_dict(), MODEL_PATH)

        cnn_trainer.evaluate()
        cnn_trainer.plot_training_curve("B/training_curve_taskB.png")

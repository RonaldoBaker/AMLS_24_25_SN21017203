# Import dependencies
from A.acquisitionA import load_breastmnist_data, analyse
from A.task_A_models import LogisticRegressionModel, KNNModel, CNNModel
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.models as models
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score


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

    # ------------------------------------------------------------------- #

    # Instantiate logistic regression model with cross-validation
    logreg_cv = LogisticRegressionModel(solver = SOLVER,
                                        with_cv = True,
                                        Cs = [0.001, 0.01, 0.1, 1, 10, 100],
                                        cv = 5,
                                        scoring = "roc_auc",
                                        max_iter = 1000)
    
    y_val_pred_cv = logreg_cv.predict(processed_train_data, processed_train_labels, processed_val_data)
    
    # TODO:This value needs to be passed directly to the roc_auc_score function, not the value above
    y_val_auc_pred = logreg_cv.model.predict_proba(processed_val_data)[:, 1]

    logreg_cv.evaluate(processed_val_labels, y_val_pred_cv)

    # ------------------------------------------------------------------- #

    # KNN model - Finding the optimum value of K (the number of nearest neighbours)
    accuracies = []
    NEIGHBOURS = 30

    for k in range(1, NEIGHBOURS+1):
        knn_model = KNNModel(neighbours=k)
        y_pred = knn_model.predict(processed_train_data, processed_train_labels, processed_test_data)
        accuracies.append(knn_model.evaluate(processed_test_labels, y_pred))

    # Plot number of nearest neighbours vs AUC-ROC accuracy
    plt.plot(range(1, NEIGHBOURS+1), accuracies, marker = 'o')
    plt.grid()
    plt.title("Accuracy vs K Value")
    plt.xlabel("No. of nearest neighbours")
    plt.ylabel("AUC-ROC Accuracy Score")
    plt.show()
 
    # ------------------------------------------------------------------- #

    # Neural networks for binary classification

    # Pre-trained model - ResNet
    # resnet18 = models.resnet18(pretrained=True)
    # TODO: finish

    # CNN from scratch
    # Define hyperparameters
    BATCH_SIZE = 16
    EPOCHS = 1000
    LEARNING_RATE = 0.001

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    cnn = CNNModel()
    cnn.to(DEVICE) 

    # Define loss function and optimiser
    loss_func = nn.BCELoss()
    optimiser = optim.Adam(cnn.parameters(), lr=LEARNING_RATE)

    # Empty list to track loss and accuracy
    train_losses = []
    batch_metrics = []
    validation_accuracies = []

    for epoch in range(EPOCHS):
        # Train in batches
        for images, labels in train_loader:
            cnn.train()
            optimiser.zero_grad()
            outputs = cnn(images)
            loss = loss_func(outputs, labels)
            batch_metrics.append(loss.item())
            loss.backward()
            optimiser.step()
        
        # Take average loss after training all the batches
        if epoch % 100 == 0:
            avg_loss = np.mean(np.array(batch_metrics))
            train_losses.append(avg_loss)
            batch_metrics.clear()

        # Evaluation
        with torch.no_grad():
            cnn.eval()
            for images, labels in val_loader:
                optimiser.zero_grad()
                outputs = cnn(images)
                predicted = (outputs > 0.5).int()
                accuracy = accuracy_score(labels, predicted)
                batch_metrics.append(accuracy)
            
            # Take average accuracy after evaluation all batches
            if epoch % 100 == 0:
                avg_accuracy = np.mean(np.array(batch_metrics))
                validation_accuracies.append(avg_accuracy)
                batch_metrics.clear()
        
        if epoch % 100 == 0:
            print(f"Epoch: {epoch} | Loss: {avg_loss: .3f} | Accuracy: {avg_accuracy: .3f}")

    # Test
    with torch.no_grad():
        for image, labels in test_loader:
            outputs = cnn(image)
            predicted_labels = (outputs > 0.5).int()
            accuracy = accuracy_score(labels, predicted_labels)
            batch_metrics.append(accuracy)

        avg_accuracy = np.mean(np.array(batch_metrics))
        print(f"Accuracy on test data: {avg_accuracy* 100: .2f}%")

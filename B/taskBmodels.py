import torch
import torch.nn as nn
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt

class EarlyStopping:
    def __init__(self, patience: int = 5, delta: int = 0):
        self.patience = patience # How many epochs to wait for improvement
        self.delta = delta # Minimum change in monitored quantity to qualify as improvement
        self.counter = 0 # Counter for patience
        self.best_loss= None # Best score so far
        self.early_stop = False # Whether to stop training

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss # Set best less to the first loss
        elif val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True  # Stop training

class CNNModel(nn.Module):
    def __init__(self):
        """
        Defines the CNN model architecture
        """
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=2, stride=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2, stride=1)  # Second Conv layer
        self.conv3 = nn.Conv2d(32, 64, kernel_size=2, stride=1)  # Third Conv layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Max Pooling
        self.relu = nn.ReLU()  # Activation function
        self.fc1 = nn.Linear(64 * 5 * 5, 512)  # Fully connected layer
        self.fc2 = nn.Linear(512, 256)  # Fully connected layer
        self.fc3 = nn.Linear(256, 128)  # Fully connected layer
        self.fc4 = nn.Linear(128, 8)  # Output layer for 8 classes
        self.dropout = nn.Dropout(0.5)  # Dropout rate
        self.softmax = nn.Softmax(dim=1)  # Activation function for multi-class classification

        """
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=2, stride=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2, stride=1)  # Second Conv layer
        self.conv3 = nn.Conv2d(32, 64, kernel_size=2, stride=1)  # Third Conv layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Max Pooling
        self.relu = nn.ReLU()  # Activation function
        self.fc1 = nn.Linear(64 * 5 * 5, 512)  # Fully connected layer
        self.fc2 = nn.Linear(512, 256)  # Fully connected layer
        self.fc3 = nn.Linear(256, 128)  # Fully connected layer
        self.fc4 = nn.Linear(128, 1)  # Single output for binary classification
        self.dropout = nn.Dropout(0.5)  # Dropout rate
        self.sigmoid = nn.Sigmoid()  # Activation function
        """

        """
        super().__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=2, stride=1)
        self.conv2 = nn.Conv2d(3, 16, kernel_size=2, stride=1)  # Second Conv layer
        self.conv3 = nn.Conv2d(16, 32, kernel_size=2, stride=1)  # Third Conv layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Max Pooling
        self.relu = nn.ReLU()  # Activation function
        self.fc1 = nn.Linear(32 * 5 * 5, 512)  # Fully connected layer
        self.fc2 = nn.Linear(512, 256)  # Fully connected layer
        self.fc3 = nn.Linear(256, 128)  # Fully connected layer
        self.fc4 = nn.Linear(128, 1)  # Single output for binary classification
        self.dropout = nn.Dropout(0.5)  # Dropout rate
        self.sigmoid = nn.Sigmoid()  # Activation function
        """
    
    def forward(self, x: ArrayLike) -> ArrayLike:
        """
        Defines how the input goes through the forward pass

        Arg:
        - x (ArrayLike): The data to pass through the neural network

        Returns:
        ArrayLike: The output of the CNN model
        """
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = x.view(x.shape[0], -1)  # Flatten for the fully connected layer
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)  # Apply softmax function for multi-class classification
        return x
    

class CNNModelTrainer:
    def __init__(
        self,
        train_data: ArrayLike,
        test_data: ArrayLike,
        val_data: ArrayLike,
        cnn_model: CNNModel,
        epochs: int,
        loss_func: torch.nn,
        optimiser: torch.optim,
    ):
        self.train_data = train_data
        self.test_data = test_data
        self.val_data = val_data
        self.cnn = cnn_model
        self.epochs = epochs
        self.loss_func = loss_func
        self.optimiser = optimiser
        # Empty lists to track loss and accuracy
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.all_predictions = []
        self.all_labels = []

    def train(self, patience: int = 5):
        # TODO: Comment and add docstring

        # Create instance of EarlyStopping class
        early_stopping = EarlyStopping(patience=patience)

        for epoch in range(self.epochs):
            running_train_loss = 0.0
            train_batch_count = 0
            running_val_loss = 0.0
            running_val_accuracy = 0.0
            val_batch_count = 0

            # Train in batches
            self.cnn.train()
            for _, (images, labels) in enumerate(self.train_data):
                self.optimiser.zero_grad()
                outputs = self.cnn(images)
                loss = self.loss_func(outputs, labels.squeeze(1).long())
                loss.backward()
                self.optimiser.step()
                running_train_loss += loss.item()
                train_batch_count += 1

            # Validation
            with torch.no_grad():
                self.cnn.eval()
                for images, labels in self.val_data:
                    self.optimiser.zero_grad()
                    outputs = self.cnn(images)
                    _, predicted = torch.max(outputs.data, 1)
                    accuracy = accuracy_score(labels.cpu(), predicted.cpu().numpy())
                    loss = self.loss_func(outputs, labels.squeeze(1).long())
                    running_val_accuracy += accuracy
                    running_val_loss += loss.item()
                    val_batch_count += 1

                    # self.all_predictions.extend(predicted.cpu())
                    # self.all_labels.extend(labels.cpu().numpy())
                    # accuracy = accuracy_score(self.all_labels, self.all_predictions)
                    # predicted = (outputs > 0.5).float()
                    # accuracy = roc_auc_score(labels.cpu(), predicted.cpu())
                    # loss = self.loss_func(outputs, labels)
                    # running_val_accuracy += accuracy
                    # running_val_loss += loss.item()
                    # val_batch_count += 1

            # Early stopping
            early_stopping(running_val_loss / val_batch_count)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch: {epoch}")
                break

            self.train_losses.append(running_train_loss / train_batch_count)
            self.val_losses.append(running_val_loss / val_batch_count)
            self.val_accuracies.append(running_val_accuracy / val_batch_count)

            if epoch % 100 == 0:
                print(f"Epoch: {epoch} | Train Loss: {self.train_losses[-1]: .3f} | Val Loss: {self.val_losses[-1]: .3f} | Val Accuracy: {self.val_accuracies[-1]: .3f}")
        print("\n")


    def evaluate(self):
        # TODO: Comment and add docstring
        all_predictions = []
        all_labels = []

        running_accuracy = 0.0
        batch_count = 0
        with torch.no_grad():
            for image, labels in self.test_data:
                outputs = self.cnn(image)
                _, predicted = torch.max(outputs.data, 1)
                accuracy = accuracy_score(labels.cpu(), predicted.cpu().numpy())
                
                
                # predicted_labels = (outputs > 0.5).int()
                # accuracy = roc_auc_score(labels.cpu(), predicted_labels.cpu())
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu())
                running_accuracy += accuracy
                batch_count += 1

        avg_accuracy = running_accuracy / batch_count
        print(f"Accuracy on test data: {avg_accuracy * 100: .2f}%\n")
        print("Classification Report (CNN)")
        print(classification_report(all_labels, all_predictions, zero_division=0))


    def plot_training_curve(self):
        # Plot the training curve
        plt.figure()
        plt.plot(range(1, len(self.train_losses)+1, 1), self.train_losses, label="Training Loss")
        plt.plot(range(1, len(self.val_losses)+1, 1), self.val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
        plt.title("CNN Training Curve")
        plt.savefig("figures/CNN_Training_Curve.png")

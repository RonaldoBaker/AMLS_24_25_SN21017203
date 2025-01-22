import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, roc_auc_score
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
        self.dropout = nn.Dropout(0.4)  # Dropout rate
        self.softmax = nn.Softmax(dim=1)  # Activation function for multi-class classification
    
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

        all_classes = list(range(8))

        for epoch in range(self.epochs):
            running_train_loss = 0.0
            train_batch_count = 0
            running_val_loss = 0.0
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
                all_labels = []
                all_probabilities = []
                for images, labels in self.val_data:
                    self.optimiser.zero_grad()
                    outputs = self.cnn(images)
                    probabilities = F.softmax(outputs, dim=1)

                    # Ensure labels are 1D
                    labels = labels.squeeze(1).long()
                    all_labels.extend(labels.cpu().numpy())
                    all_probabilities.extend(probabilities.cpu().numpy())
                    
                    # Compute loss
                    loss = self.loss_func(outputs, labels)
                    
                    running_val_loss += loss.item()
                    val_batch_count += 1

            all_labels = np.array(all_labels)
            all_probabilities = np.array(all_probabilities)

            # Compute ROC AUC score
            accuracy = roc_auc_score(all_labels, all_probabilities, multi_class='ovr', labels=all_classes)

            # Early stopping
            early_stopping(running_val_loss / val_batch_count)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch: {epoch}")
                break

            self.train_losses.append(running_train_loss / train_batch_count)
            self.val_losses.append(running_val_loss / val_batch_count)
            self.val_accuracies.append(accuracy)

            if epoch % 100 == 0:
                print(f"Epoch: {epoch} | Train Loss: {self.train_losses[-1]: .3f} | Val Loss: {self.val_losses[-1]: .3f} | Val Accuracy: {self.val_accuracies[-1]: .3f}")
        print("\n")


    def evaluate(self):
        # TODO: Comment and add docstring
        all_predictions = []
        all_labels = []

        self.cnn.eval()
        with torch.no_grad():
            all_labels = []
            all_probabilities = []
            for image, labels in self.test_data:
                outputs = self.cnn(image)
                probabilities = F.softmax(outputs, dim=1)

                # Ensure labels are 1D
                labels = labels.squeeze(1).long()
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        all_probabilities = np.array(all_probabilities)
        all_predictions = torch.argmax(torch.from_numpy(all_probabilities), dim=1).cpu().numpy()
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        # Compute ROC AUC score
        all_classes = list(range(8))
        accuracy = roc_auc_score(all_labels, all_probabilities, multi_class='ovr', labels=all_classes)   

        print(f"ROC-AUC Score: {accuracy * 100: .2f}%\n")
        print("Classification Report (CNN)")
        print(classification_report(all_labels, all_predictions, zero_division=0))


    def plot_training_curve(self, filepath: str):
        # Plot the training curve
        plt.figure(figsize=(6,4))
        plt.plot(range(1, len(self.train_losses)+1, 1), self.train_losses, label="Training Loss", linewidth=2)
        plt.plot(range(1, len(self.val_losses)+1, 1), self.val_losses, label="Validation Loss", linewidth=2, linestyle='--')
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.8)
        plt.title("Training and Validation Loss for Task B CNN", fontsize=12)
        plt.tight_layout()
        plt.savefig(filepath)

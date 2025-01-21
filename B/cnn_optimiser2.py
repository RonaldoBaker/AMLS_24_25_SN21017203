import optuna
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import List
from numpy.typing import ArrayLike
from sklearn.metrics import accuracy_score, roc_auc_score
from acquisitionB import load_bloodmnist_data
from taskBmodels import EarlyStopping
from preprocessingB import preprocess_for_cnn

# Set device
if torch.cuda.is_available():
    DEVICE_NUM = 1
    torch.cuda.set_device(DEVICE_NUM)
    DEVICE = torch.device(f"cuda:{DEVICE_NUM}")
else:
    DEVICE = torch.device("cpu")


class CNNModel(nn.Module):
    def __init__(self):
        """
        Defines the CNN model architecture
        """
        super().__init__()
        # self.conv1 = nn.Conv2d(1, 3, kernel_size=2, stride=2) # First Conv layer
        # self.conv2 = nn.Conv2d(3, 16, kernel_size=2, stride=2) # Second Conv layer
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # Max Pooling
        # self.relu = nn.ReLU() # Activation function
        # self.fc1 = nn.Linear(16 * 1 * 1, 8) # Fully connected layer
        # self.fc2 = nn.Linear(8, 1) # Single output for binary classification
        # self.sigmoid = nn.Sigmoid() # Activation function

        self.conv1 = nn.Conv2d(3, 16, kernel_size=2, stride=1) # First Conv layer
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2, stride=1) # Second Conv layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # Max Pooling
        self.relu = nn.ReLU() # Activation function
        self.fc1 = nn.Linear(32 * 6 * 6, 512) # Fully connected layer
        self.fc2 = nn.Linear(512, 256) # Fully connected layer
        self.fc3 = nn.Linear(256, 128) # Fully connected layer
        self.fc4 = nn.Linear(128, 8) # Output layer for 8 classes
        self.softmax = nn.Softmax(dim=1) # Activation function

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
        x = x.view(x.shape[0], -1) # Flatten for the fully connected layer
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x)) 
        x = self.relu(self.fc3(x))
        x = self.fc4(x) # Apply softmax function for multi-class classification
        return x


def get_dataloaders(datasets: List[ArrayLike], batch_size: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates DataLoaders for the train, test, and validation datasets.

    Arg(s):
    - datasets (List[ArrayLike]): The datasets to be converted into DataLoaders
    - batch_size (int): The batch size for the DataLoaders

    Returns:
    - List[DataLoader]: The DataLoaders for the train, test, and validation datasets
    """
    dataloaders = []
    for dataset in datasets:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        dataloaders.append(loader)
    return dataloaders[0], dataloaders[1], dataloaders[2]


def objective(trial, datasets: List[ArrayLike]):
    train_losses = []
    val_losses = []
    val_accuracies = []

    # Create hyperparameter search space
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
    # dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.1)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])

    # Get dataloaders
    train_loader, test_loader, val_loader = get_dataloaders(datasets, batch_size)

    # Instantiate model, loss function and optimiser
    cnn = CNNModel()
    cnn.to(DEVICE)
    loss_func = nn.CrossEntropyLoss()
    optimiser = optim.Adam(cnn.parameters(), lr=learning_rate)

    # Early stopping setup
    early_stopping = EarlyStopping(patience=3)

    # Training loop
    epochs = 500
    for epoch in tqdm(range(epochs)):
        running_train_loss = 0.0
        train_batch_count = 0
        running_val_loss = 0.0
        running_val_accuracy = 0.0
        val_batch_count = 0

        # Train in batches
        cnn.train()
        for images, labels in train_loader:
            optimiser.zero_grad()
            outputs = cnn(images)
            loss = loss_func(outputs, labels.squeeze(1).long())
            loss.backward()
            optimiser.step()
            running_train_loss += loss.item()
            train_batch_count += 1

            # Validation
            with torch.no_grad():
                cnn.eval()
                for images, labels in val_loader:
                    optimiser.zero_grad()
                    outputs = cnn(images)
                    probabilities = F.softmax(outputs, dim=1)

                    # Ensure labels are 1D
                    labels = labels.squeeze(1).long()
                    
                    # Compute loss
                    loss = loss_func(outputs, labels)
                    # running_val_accuracy += accuracy
                    running_val_loss += loss.item()
                    val_batch_count += 1

            # Compute ROC AUC score
            accuracy = roc_auc_score(labels.cpu(), probabilities.cpu(), multi_class='ovr', labels=[0, 1, 2, 3, 4, 5, 6, 7])

        train_losses.append(running_train_loss / train_batch_count)
        val_losses.append(running_val_loss / val_batch_count)
        # val_accuracies.append(running_val_accuracy / val_batch_count)
        val_accuracies.append(accuracy)

        if epoch % 100 == 0:
            print(f"Epoch: {epoch} | Train Loss: {train_losses[-1]: .3f} | Val Loss: {val_losses[-1]: .3f} | Val Accuracy: {val_accuracies[-1]: .3f}")

        # Using optuna pruning
        val_accuracy = val_accuracies[-1]
        trial.report(val_accuracy, epoch)

        # Prune unpromising trials
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        # Early stopping check
        early_stopping(running_val_loss / val_batch_count)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch: {epoch}")
            break

    return val_accuracy

# Prepare data
# Load BloodMNIST data
data = load_bloodmnist_data(datapath="Datasets/bloodmnist.npz")

# Separate data
train_data = data["train_data"]
train_labels = data["train_labels"]
test_data = data["test_data"]
test_labels = data["test_labels"]
val_data = data["val_data"]
val_labels = data["val_labels"]

# Reshape data to be channel first
train_data = train_data.transpose(0, 3, 1, 2)
test_data = test_data.transpose(0, 3, 1, 2)
val_data = val_data.transpose(0, 3, 1, 2)

data = preprocess_for_cnn([train_data, test_data, val_data])
train_data, test_data, val_data = data[0], data[1], data[2]

# Create tensors and add dimension for greyscale image data, and make labels 2D
train_data_tensor = torch.tensor(train_data, device=DEVICE, dtype=torch.float32)
train_labels_tensor = torch.tensor(train_labels, device=DEVICE, dtype=torch.float32)

test_data_tensor = torch.tensor(test_data, device=DEVICE, dtype=torch.float32)
test_labels_tensor = torch.tensor(test_labels, device=DEVICE, dtype=torch.float32)

val_data_tensor = torch.tensor(val_data, device=DEVICE, dtype=torch.float32)  
val_labels_tensor = torch.tensor(val_labels, device=DEVICE, dtype=torch.float32)

train_set = [(train_data_tensor[i], train_labels_tensor[i]) for i in range(len(train_data_tensor))]
test_set = [(test_data_tensor[i], test_labels_tensor[i]) for i in range(len(test_data_tensor))]
val_set = [(val_data_tensor[i], val_labels_tensor[i]) for i in range(len(val_data_tensor))]

# Set up Optuna Study with Median Pruner
pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20, interval_steps=5)
study = optuna.create_study(direction="maximize", pruner=pruner)

# Run the optimisation
study.optimize(lambda trial: objective(trial, [train_set, test_set, val_set]), n_trials=60)

# Print the best hyperparameters
print("Best hyperparameters:", study.best_params)
print("Best value:", study.best_value)
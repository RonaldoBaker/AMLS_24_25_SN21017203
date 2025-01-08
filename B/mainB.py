import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from B.acquisitionB import load_bloodmnist_data, display_info
from B.taskBmodels import CNNModel, CNNModelTrainer

def taskB():
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

    # CNN model from Task A
    BATCH_SIZE = 64
    EPOCHS = 1000
    LEARNING_RATE = 0.001
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
    print("CNN\n")
    cnn = CNNModel()
    cnn.to(DEVICE) 

    # Define loss function and optimiser
    loss_func = nn.CrossEntropyLoss()
    optimiser = optim.Adam(cnn.parameters(), lr=LEARNING_RATE)

    # Train model
    cnn_trainer = CNNModelTrainer(train_loader, test_loader, val_loader, cnn, EPOCHS, loss_func, optimiser)
    cnn_trainer.train(patience=3)
    cnn_trainer.evaluate()
    cnn_trainer.plot_training_curve()

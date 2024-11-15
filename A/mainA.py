# Import dependencies
from acquisitionA import load_breastmnist_data

# Define constant(s)
DATAPATH = "Datasets/breastmnist.npz"

def taskA():
    """
    Executes task A, including data loading and model training/evaluation
    """
    # Load BreastMNIST data
    breastmnist_data = load_breastmnist_data(datapath=DATAPATH)
    print(breastmnist_data)
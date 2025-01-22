import numpy as np
from numpy.typing import ArrayLike
from typing import Dict
from medmnist import BloodMNIST
from rich import print as rprint

def load_bloodmnist_data(datapath: str) -> Dict[str, ArrayLike]:
    """
    Loads the compressed BloodMNIST dataset from MedMNIST which is saved locally and returns a list of numpy arrays from the dataset.
    
    Arg:
    - datapath (str): relative or absolute path to stored dataset

    Returns:
    - Dict[str, ArrayLike]: A dictionary of (string, numpy array) pairs
    """
    with np.load(datapath) as breastmnist_data:
        # Image data
        train_data = breastmnist_data["train_images"]
        val_data = breastmnist_data["val_images"]
        test_data = breastmnist_data["test_images"]

        # Label data
        train_labels = breastmnist_data["train_labels"]
        val_labels = breastmnist_data["val_labels"]
        test_labels = breastmnist_data["test_labels"]

    return {"train_data": train_data,
            "train_labels": train_labels,
            "val_data": val_data,
            "val_labels": val_labels, 
            "test_data": test_data,
            "test_labels": test_labels}

def display_info(data: ArrayLike):
    # Get dataset info
    dataset = BloodMNIST(split="train", download=True)
    rprint(dataset.info, "\n")

    # Find shape of arrays
    print(f"The shape of the training data is {data['train_data'].shape}")
    print(f"The shape of the validation data is {data['val_data'].shape}")
    print(f"The shape of the testing data is {data['test_data'].shape}\n")

    # Find how many of each class exists
    data_labels = np.concatenate((data["train_labels"], data["test_labels"], data["val_labels"]), axis = 0)
    _, count = np.unique(data_labels, return_counts = True)
    print(f"There are {count[0]} samples of class '0'")
    print(f"There are {count[1]} samples of class '1'")
    print(f"There are {count[2]} samples of class '2'")
    print(f"There are {count[3]} samples of class '3'")
    print(f"There are {count[4]} samples of class '4'")
    print(f"There are {count[5]} samples of class '5'")
    print(f"There are {count[6]} samples of class '6'")
    print(f"There are {count[7]} samples of class '7'")
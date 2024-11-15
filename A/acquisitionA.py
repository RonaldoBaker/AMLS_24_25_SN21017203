# Import dependencies
import numpy as np
from numpy.typing import ArrayLike
from typing import List

def load_breastmnist_data(datapath: str) -> List[ArrayLike]:
    """
    Loads the compressed BreastMNIST dataset from MedMNIST which is saved locally and returns a list of numpy arrays from the dataset. 

    Args:
    - datapath (str): relative or absolute path to stored dataset

    Returns:
    - List[ArrayLike]: A list of all numpy arrays extracted from the compressed numpy dataset
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

    return [train_data, train_labels, val_data, val_labels, train_data, train_labels]


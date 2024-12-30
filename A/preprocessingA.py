# Import dependencies
import numpy as np
from numpy.typing import ArrayLike
from typing import List
from imblearn.over_sampling import SMOTE

def preprocess_for_traditional(data: tuple[ArrayLike, ArrayLike], labels: ArrayLike) -> ArrayLike:
    """
    Prepares data for traditional machine learning models by reshaping and normalising pixel values.
    The training dataset is then balanced using SMOTE.

    Arg(s):
    - data (List[ArrayLike]): The data to be preprocessed
    - labels (List[ArrayLike]): The labels of the data to be preprocessed

    Returns:
    - tuple[List[ArrayLike], List[ArrayLike]]: The preprocessed train and test data and labels
    """
    for i in range(len(data)):
        # Reshape data from 3D to 2D numpy arrays for traditional machine learning models
        data[i] = data[i].reshape(data[i].shape[0], -1)

        # Normalise pixel values to (0, 1)
        data[i] = data[i] / 255.0

    for i in range(len(labels)):
        # Flatten labels
        labels[i] = np.ravel(labels[i])

    # Balance the training dataset using the Synthetic Minority Over-sampling Technique (SMOTE)
    smote = SMOTE(random_state=7)
    data[0], labels[0] = smote.fit_resample(data[0], labels[0])

    return data, labels

def preprocess_for_cnn(data: tuple[ArrayLike, ArrayLike], labels: ArrayLike) -> ArrayLike:
    """
    Prepares data for traditional machine learning models by reshaping and normalising pixel values.
    The training dataset is then balanced using SMOTE.

    Arg(s):
    - data (List[ArrayLike]): The data to be preprocessed
    - labels (List[ArrayLike]): The labels of the data to be preprocessed

    Returns:
    - tuple[List[ArrayLike], List[ArrayLike]]: The preprocessed train and test data and labels
    """
    # Normalise pixel values to (0, 1)
    for i in range(len(data)):
        # Reshape data from 3D to 2D numpy arrays for traditional machine learning models
        # data[i] = data[i].reshape(data[i].shape[0], -1)

        # Normalise pixel values to (0, 1)
        data[i] = data[i] / 255.0
       

    # # Flatten training data and training labels for SMOTE
    # flattened_train_data = data[0].reshape(data[0].shape[0], -1)
    # flattened_train_labels = np.ravel(labels[0])

    # # Balance the training dataset using the Synthetic Minority Over-sampling Technique (SMOTE)
    # smote = SMOTE(random_state=7)
    # data[0], labels[0] = smote.fit_resample(flattened_train_data, flattened_train_labels)

    # Reshape data for CNN
    for i in range(len(data)):
        data[i] = data[i].reshape(data[i].shape[0], 1, 28, 28)

    # # Reshape training labels after SMOTE
    # labels[0] = labels[0].reshape(labels[0].shape[0], 1)

    return data, labels


def balance_data(data: ArrayLike, labels: ArrayLike) -> ArrayLike:
    """
    Balances the dataset using the Synthetic Minority Over-sampling Technique (SMOTE).

    Arg(s):
    - data (ArrayLike): The data to be balanced
    - labels (ArrayLike): The labels of the data to be balanced

    Returns:
    - tuple[ArrayLike, ArrayLike]: The balanced data and labels
    """
    smote = SMOTE(random_state=7)
    data_resampled, labels_resampled = smote.fit_resample(data, labels)
    return data_resampled, labels_resampled

def scale(data: List[ArrayLike]) -> List[ArrayLike]:
    """
    Scales the data to the range [0, 1].

    Arg(s):
    - data (List[ArrayLike]): The data to be scaled

    Returns:
    - List[ArrayLike]: The scaled data
    """
    for i in range(len(data)):
        data[i] = data[i] / 255.0
    return data

def flatten_labels(labels: List[ArrayLike]) -> List[ArrayLike]:
    """
    Flattens the labels to 1D arrays.
    """
    for i in range(len(labels)):
        labels[i] = np.ravel(labels[i])
    return labels
    
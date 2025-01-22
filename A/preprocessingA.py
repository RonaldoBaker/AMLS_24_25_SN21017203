# Import dependencies
import numpy as np
from numpy.typing import ArrayLike
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

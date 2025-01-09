from numpy.typing import ArrayLike

def preprocess_for_traditional(data: tuple[ArrayLike, ArrayLike]) -> ArrayLike:
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
        # Reshape each array from 3D to 2D numpy arrays for traditional machine learning models
        data[i] = data[i].reshape(data[i].shape[0], -1)

        # Normalize pixel values to the range [0, 1]
        data[i] = data[i] / 255.0

    return data
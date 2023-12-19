import numpy as np
import torch

from sklearn.datasets import fetch_openml
from torch import Tensor
from torch.nn.modules.loss import _Loss


def nll(outputs: Tensor, y: Tensor) -> Tensor:
    """Implementation of the negative log-likelihood loss with PyTorch.

    Args:
        outputs (Tensor): model's output tensor
        y (Tensor): target tensor

    Returns:
        Tensor: loss tensor
    """
    
    return -torch.log(outputs.gather(1, y.view(-1, 1)))

def cross_entropy(outputs: Tensor, y: Tensor) -> Tensor:
    """Implementation of the cross-entropy loss with PyTorch.

    Args:
        outputs (Tensor): model's output tensor
        y (Tensor): target tensor

    Returns:
        Tensor: loss tensor
    """
    
    return -torch.log_softmax(outputs, dim=1).gather(1, y.view(-1, 1))

def softmax(x: Tensor) -> Tensor:
    """Implementation of the softmax function with PyTorch.

    Args:
        x (Tensor): input tensor

    Returns:
        Tensor: output tensor
    """
    
    return torch.exp(x) / torch.exp(x).sum(dim=1, keepdim=True)



def data_mnist(train_start=0, train_end=60000, test_start=0, test_end=10000):
    """
    Load and preprocess MNIST dataset using sklearn
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :return: tuple of four tensors containing training data, training labels,
             testing data and testing labels.
    """
    assert isinstance(train_start, int)
    assert isinstance(train_end, int)
    assert isinstance(test_start, int)
    assert isinstance(test_end, int)

    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    X, Y = mnist["data"], mnist["target"]

    # Convert DataFrame to numpy array
    X = X.to_numpy()

    # Convert to tensors
    X = torch.tensor(X / 255., dtype=torch.float32)
    Y = torch.tensor(Y.astype(int), dtype=torch.int64)

    # Split into training and test sets
    X_train = X[train_start:train_end]
    Y_train = Y[train_start:train_end]
    X_test = X[test_start:test_end]
    Y_test = Y[test_start:test_end]

    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)

    return X_train, Y_train, X_test, Y_test

import numpy as np
from .base import Criterion
from .activations import LogSoftmax


class MSELoss(Criterion):
    """
    Mean squared error criterion
    """
    def compute_output(self, input: np.array, target: np.array) -> float:
        """
        :param input: array of size (batch_size, *)
        :param target:  array of size (batch_size, *)
        :return: loss value
        """
        assert input.shape == target.shape, 'input and target shapes not matching'
        
        return np.mean((input - target) ** 2)

    def compute_grad_input(self, input: np.array, target: np.array) -> np.array:
        """
        :param input: array of size (batch_size, *)
        :param target:  array of size (batch_size, *)
        :return: array of size (batch_size, *)
        """
        assert input.shape == target.shape, 'input and target shapes not matching'
        
        B, N = input.shape
        return 2 * (input - target) / (B * N)


class CrossEntropyLoss(Criterion):
    """
    Cross-entropy criterion over distribution logits
    """
    def __init__(self):
        super().__init__()
        self.log_softmax = LogSoftmax()

    def compute_output(self, input: np.array, target: np.array) -> float:
        """
        :param input: logits array of size (batch_size, num_classes)
        :param target: labels array of size (batch_size, )
        :return: loss value
        """
        B, C = input.shape
        classes = np.arange(C)
        return -np.sum((classes == target.reshape(B, 1)) * self.log_softmax(input)) / B


    def compute_grad_input(self, input: np.array, target: np.array) -> np.array:
        """
        :param input: logits array of size (batch_size, num_classes)
        :param target: labels array of size (batch_size, )
        :return: array of size (batch_size, num_classes)
        """
        B, C = input.shape
        classes = np.arange(C)
        grad_log_soft_max = (classes == target.reshape(B, 1)) / -B
        return self.log_softmax.compute_grad_input(input, grad_log_soft_max)

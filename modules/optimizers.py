import numpy as np
from typing import Tuple
from .base import Module, Optimizer


class SGD(Optimizer):
    """
    Optimizer implementing stochastic gradient descent with momentum
    """
    def __init__(self, module: Module, lr: float = 1e-2, momentum: float = 0.0,
                 weight_decay: float = 0.0):
        """
        :param module: neural network containing parameters to optimize
        :param lr: learning rate
        :param momentum: momentum coefficient (alpha)
        :param weight_decay: weight decay (L2 penalty)
        """
        super().__init__(module) self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

    def step(self):
        parameters = self.module.parameters()
        gradients = self.module.parameters_grad()
        if 'm' not in self.state:
            self.state['m'] = [np.zeros_like(param) for param in parameters]

        for param, grad, m in zip(parameters, gradients, self.state['m']):
            """
              - update momentum variable (m)
              - update parameter variable (param)
              we need to change original array, not its copy
            """
            np.add(grad, self.weight_decay * param, out=grad)
            np.add(m * self.momentum, grad, out=m)
            np.add(param, -self.lr * m, out=param)


class Adam(Optimizer):
    """
    Optimizer implementing Adam
    """
    def __init__(self, module: Module, lr: float = 1e-3,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0):
        """
        :param module: neural network containing parameters to optimize
        :param lr: learning rate
        :param betas: Adam beta1 and beta2
        :param eps: Adam eps
        :param weight_decay: weight decay (L2 penalty)
        """
        super().__init__(module)
        self.lr = lr
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.weight_decay = weight_decay

    def step(self):
        parameters = self.module.parameters()
        gradients = self.module.parameters_grad()
        if 'm' not in self.state:
            self.state['m'] = [np.zeros_like(param) for param in parameters]
            self.state['v'] = [np.zeros_like(param) for param in parameters]
            self.state['t'] = 0

        self.state['t'] += 1
        t = self.state['t']
        for param, grad, m, v in zip(parameters, gradients, self.state['m'], self.state['v']):
            """
              - update first moment variable (m)
              - update second moment variable (v)
              - update parameter variable (param)
              we need to change original array, not its copy
            """
            np.add(grad, self.weight_decay * param, out=grad)
            np.add(m * self.beta1, (1 - self.beta1) * grad, out=m)
            np.add(v * self.beta2, (1 - self.beta2) * grad ** 2, out=v)
            norm_m = m / (1 - self.beta1 ** t)
            norm_v = v / (1 - self.beta2 ** t)
            np.add(param, -self.lr * norm_m / (np.sqrt(norm_v) + self.eps), out=param)

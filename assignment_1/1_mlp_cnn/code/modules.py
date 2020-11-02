"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np


class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """

    def __init__(self, in_features, out_features):
        """
        Initializes the parameters of the module.

        Args:
          in_features: size of each input sample
          out_features: size of each output sample

        TODO:
        Initialize weights self.params['weight'] using normal distribution with mean = 0 and
        std = 0.0001. Initialize biases self.params['bias'] with 0.

        Also, initialize gradients with zeros.
        """

        self.params = {}
        self.grads = {}
        self.__values = {}

        self.params["weight"] = np.random.normal(
            0, 0.0001, size=(in_features, out_features)
        )
        self.params["bias"] = np.zeros(out_features, dtype=float)

        for key, val in self.params.items():
            self.grads[key] = np.zeros_like(val)

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        multiplication = x @ self.params["weight"]
        out = multiplication + self.params["bias"]

        self.__values["x"] = x
        self.__values["weight"] = self.params["weight"]
        # self.__values['multiplication'] = multiplication
        # self.__values['out'] = out
        return out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to
        layer parameters in self.grads['weight'] and self.grads['bias'].
        """

        self.grads["weight"] = (dout.T @ self.__values["x"]).T
        self.grads["bias"] = (dout.T @ np.ones(dout.shape[0])).T
        dx = dout @ self.__values["weight"].T

        return dx


class SoftMaxModule(object):
    """
    Softmax activation module.
    """

    def forward(self, x):
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.
        To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        b = x.max(axis=1)
        out = np.exp(x - b[:, np.newaxis])
        out = out / out.sum(axis=1)[:, np.newaxis]

        self.__x = x
        self.__out = out

        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous modul
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        # FIX THIS
        dx = np.zeros(self.__x.shape)
        for i in range(self.__x.shape[0]):
            for j in range(self.__x.shape[1]):
                dx_ij = [
                    dout[i, l]
                    * self.__out[i, l]
                    * (int(j == l) - self.__out[i, j])
                    for l in range(dout.shape[1])
                ]
                dx[i, j] = np.sum(dx_ij)

        return dx


class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """

    def forward(self, x, y):
        """
        Forward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          out: cross entropy loss

        TODO:
        Implement forward pass of the module.
        """

        S = y.shape[0]
        out = -(1 / S) * np.sum(y * np.log(x))
        return out

    def backward(self, x, y):
        """
        Backward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          dx: gradient of the loss with the respect to the input x.

        TODO:
        Implement backward pass of the module.
        """

        S = y.shape[0]
        dx = -(1 / S) * (y / x)
        return dx


class ELUModule(object):
    """
    ELU activation module.
    """

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        mask = x < 0
        out = np.zeros_like(x)
        self.__mask = mask
        self.__exp_masked = np.exp(x[mask])
        out[mask] = self.__exp_masked - 1
        out[~mask] = x[~mask]

        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        dx = np.empty(self.__mask.shape)
        dx[self.__mask] = self.__exp_masked
        dx[~self.__mask] = 1

        dx = dx * dout

        return dx

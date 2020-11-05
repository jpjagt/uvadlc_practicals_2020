import numpy as np
import torch
import torch.nn as nn

"""
The modules/function here implement custom versions of layer normalization in PyTorch.
In contrast to more advanced implementations no use of a running mean/variance is made.
You should fill in code into indicated sections.
"""


######################################################################################
# Code for Question 3.1
######################################################################################


class CustomLayerNormAutograd(nn.Module):
    """
    This nn.module implements a custom version of the layer norm operation for MLPs.
    The operations called in self.forward track the history if the input tensors have the
    flag requires_grad set to True. The backward pass does not need to be implemented, it
    is dealt with by the automatic differentiation provided by PyTorch.
    """

    def __init__(self, n_neurons, eps=1e-5):
        """
        Initializes CustomLayerNormAutograd object.

        Args:
          n_neurons: int specifying the number of neurons
          eps: small float to be added to the variance for stability

        TODO:
          Save parameters for the number of neurons and eps.
          Initialize parameters gamma and beta via nn.Parameter
        """
        super(CustomLayerNormAutograd, self).__init__()

        # self.n_neurons = torch.Tensor(n_neurons)
        # self.eps = torch.Tensor(eps)
        self.n_neurons = n_neurons
        self.eps = eps

        std = 5e-8
        self.gamma = nn.Parameter(
            # torch.zeros(self.n_neurons)
            torch.normal(torch.zeros(self.n_neurons), std)
        )
        self.beta = nn.Parameter(
            torch.normal(torch.zeros(self.n_neurons), std)
        )
        # nn.init.kaiming_uniform_(self.gamma, a=math.sqrt(5))
        # nn.init.kaiming_uniform_(self.beta, a=math.sqrt(5))

    def forward(self, input):
        """
        Compute the layer normalization

        Args:
          input: input tensor of shape (n_batch, n_neurons)
        Returns:
          out: layer-normalized tensor

        TODO:
          Check for the correctness of the shape of the input tensor.
          Implement layer normalization forward pass as given in the assignment.
          For the case that you make use of torch.var be aware that the flag unbiased=False should be set.
        """

        if len(input.size()) > 2:
            raise ValueError("Input should be two-dimensional")
        if input.size()[1] != self.n_neurons:
            raise ValueError(
                "n_neurons does not match size of second dimension of input"
            )

        var, mean = torch.var_mean(input, dim=1, unbiased=False)
        input_norm = ((input.T - mean) / torch.sqrt(var + self.eps)).T
        out = self.gamma * input_norm + self.beta
        return out


######################################################################################
# Code for Question 3.2 b)
######################################################################################


class CustomLayerNormManualFunction(torch.autograd.Function):
    """
    This torch.autograd.Function implements a functional custom version of the layer norm operation for MLPs.
    Using torch.autograd.Function allows you to write a custom backward function.
    The function will be called from the nn.Module CustomLayerNormManualModule
    Inside forward the tensors are (automatically) not recorded for automatic differentiation since the backward
    pass is done via the backward method.
    The forward pass is not called directly but via the apply() method. This makes sure that the context objects
    are dealt with correctly. Example:
      my_bn_fct = CustomLayerNormManualFunction()
      normalized = fct.apply(input, gamma, beta, eps)
    """

    @staticmethod
    def forward(ctx, input, gamma, beta, eps=1e-5):
        """
        Compute the layer normalization

        Args:
          ctx: context object handling storing and retrival of tensors and constants and specifying
               whether tensors need gradients in backward pass
          input: input tensor of shape (n_batch, n_neurons)
          gamma: variance scaling tensor, applied per neuron, shpae (n_neurons)
          beta: mean bias tensor, applied per neuron, shpae (n_neurons)
          eps: small float added to the variance for stability
        Returns:
          out: layer-normalized tensor

        TODO:
          Implement the forward pass of layer normalization
          Store constant non-tensor objects via ctx.constant=myconstant
          Store tensors which you need in the backward pass via ctx.save_for_backward(tensor1, tensor2, ...)
          Intermediate results can be decided to be either recomputed in the backward pass or to be stored
          for the backward pass. Do not store tensors which are unnecessary for the backward pass to save memory!
          For the case that you make use of torch.var be aware that the flag unbiased=False should be set.
        """

        var, mean = torch.var_mean(input, dim=1, unbiased=False)
        input_norm = ((input.T - mean) / torch.sqrt(var + eps)).T
        out = gamma * input_norm + beta

        ctx.save_for_backward(input, gamma, beta, var, mean, input_norm)
        ctx.eps = eps
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        Compute backward pass of the layer normalization.

        Args:
          ctx: context object handling storing and retrival of tensors and constants and specifying
               whether tensors need gradients in backward pass
        Returns:
          out: tuple containing gradients for all input arguments

        TODO:
          Retrieve saved tensors and constants via ctx.saved_tensors and ctx.constant
          Compute gradients for inputs where ctx.needs_input_grad[idx] is True. Set gradients for other
          inputs to None. This should be decided dynamically.
        """
        grad_beta = grad_gamma = grad_input = None

        input, gamma, beta, var, mean, input_norm = ctx.saved_tensors
        gamma = gamma.view(-1, 1)
        mean = mean.view(-1, 1)
        var = var.view(-1, 1)
        eps = ctx.eps
        S, M = grad_output.size()

        if ctx.needs_input_grad[0]:
            dy_mul_gamma = (grad_output @ gamma).view(-1, 1)
            ve = var + eps
            ve_half_inv = ve ** (-1 / 2)
            ve_three_over_two_inv = ve ** (-3 / 2)
            grad_input = ve_half_inv * (
                grad_output * gamma.T - 1 / M * dy_mul_gamma
            ) + (
                -1
                / M
                * ve_three_over_two_inv
                * (
                    (dy_mul_gamma) * (-input * mean + mean * mean)
                    + ((grad_output * input) @ gamma) * (input - mean)
                )
            )

        if ctx.needs_input_grad[1]:
            grad_gamma = (grad_output * input_norm).T @ torch.ones(
                S, dtype=torch.float64
            )
        if ctx.needs_input_grad[2]:
            grad_beta = grad_output.T @ torch.ones(S, dtype=torch.float64)

        return grad_input, grad_gamma, grad_beta, None


######################################################################################
# Code for Question 3.2 c)
######################################################################################


class CustomLayerNormManualModule(nn.Module):
    """
    This nn.module implements a custom version of the layer norm operation for MLPs.
    In self.forward the functional version CustomLayerNormManualFunction.forward is called.
    The automatic differentiation of PyTorch calls the backward method of this function in the backward pass.
    """

    def __init__(self, n_neurons, eps=1e-5):
        """
        Initializes CustomLayerNormManualModule object.

        Args:
          n_neurons: int specifying the number of neurons
          eps: small float to be added to the variance for stability

        TODO:
          Save parameters for the number of neurons and eps.
          Initialize parameters gamma and beta via nn.Parameter
        """
        super(CustomLayerNormManualModule, self).__init__()

        self.n_neurons = n_neurons
        self.eps = eps

        std = 5e-6
        self.gamma = nn.Parameter(
            torch.normal(torch.zeros(self.n_neurons), std)
        )
        self.beta = nn.Parameter(
            torch.normal(torch.zeros(self.n_neurons), std)
        )

    def forward(self, input):
        """
        Compute the layer normalization via CustomLayerNormManualFunction

        Args:
          input: input tensor of shape (n_batch, n_neurons)
        Returns:
          out: layer-normalized tensor

        TODO:
          Check for the correctness of the shape of the input tensor.
          Instantiate a CustomLayerNormManualFunction.
          Call it via its .apply() method.
        """

        if len(input.size()) > 2:
            raise ValueError("Input should be two-dimensional")
        if input.size()[1] != self.n_neurons:
            raise ValueError(
                "n_neurons does not match size of second dimension of input"
            )

        custom_layer_norm_manual = CustomLayerNormManualFunction()
        out = custom_layer_norm_manual.apply(
            input, self.gamma, self.beta, self.eps
        )
        return out


if __name__ == "__main__":
    # create test batch
    n_batch = 8
    n_neurons = 128
    # create random tensor with variance 2 and mean 3
    x = 2 * torch.randn(n_batch, n_neurons, requires_grad=True) + 10
    print(
        "Input data:\n\tmean={}\n\tvars={}".format(
            x.mean(dim=1).data, x.var(dim=1).data
        )
    )

    # test CustomLayerNormAutograd
    print("3.1) Test automatic differentation version")
    bn_auto = CustomLayerNormAutograd(n_neurons)
    y_auto = bn_auto(x)
    print(
        "\tmeans={}\n\tvars={}".format(
            y_auto.mean(dim=1).data, y_auto.var(dim=1).data
        )
    )

    # test CustomLayerNormManualFunction
    # this is recommended to be done in double precision
    print("3.2 b) Test functional version")
    input = x.double()
    gamma = torch.sqrt(
        10 * torch.arange(n_neurons, dtype=torch.float64, requires_grad=True)
    )
    beta = 100 * torch.arange(
        n_neurons, dtype=torch.float64, requires_grad=True
    )
    bn_manual_fct = CustomLayerNormManualFunction(n_neurons)
    y_manual_fct = bn_manual_fct.apply(input, gamma, beta)
    print(
        "\tmeans={}\n\tvars={}".format(
            y_manual_fct.mean(dim=1).data, y_manual_fct.var(dim=1).data
        )
    )
    # gradient check
    grad_correct = torch.autograd.gradcheck(
        bn_manual_fct.apply, (input, gamma, beta)
    )
    if grad_correct:
        print("\tgradient check successful")
    else:
        raise ValueError("gradient check failed")

    # test CustomLayerNormManualModule
    print("3.2 c) Test module of functional version")
    bn_manual_mod = CustomLayerNormManualModule(n_neurons)
    y_manual_mod = bn_manual_mod(x)
    print(
        "\tmeans={}\n\tvars={}".format(
            y_manual_mod.mean(dim=1).data, y_manual_mod.var(dim=1).data
        )
    )

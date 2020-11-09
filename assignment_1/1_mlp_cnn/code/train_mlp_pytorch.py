"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
import argparse
import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils
import time
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.nn as nn

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = "100"
LEARNING_RATE_DEFAULT = 1e-3
MAX_STEPS_DEFAULT = 1400
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100


# my defaults :D
OPTIMIZER_DEFAULT = "Adam"
OPTIMIZER_MOMENTUM_DEFAULT = 0.9
ACTIVATION_FN_DEFAULT = "ELU"
SCHEDULER_DEFAULT = "StepLR"
SCHEDULER_STEP_SIZE_DEFAULT = MAX_STEPS_DEFAULT // 3
SCHEDULER_GAMMA_DEFAULT = 0.5

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = "./cifar10/cifar-10-batches-py"

FLAGS = None

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_scheduler(name, optimizer):
    print("gam", FLAGS.scheduler_gamma)
    scheduler = None
    if name == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=FLAGS.scheduler_step_size,
            gamma=FLAGS.scheduler_gamma,
        )
    return scheduler


def get_optimizer(name, params):
    optimizers = {
        "SGD": torch.optim.SGD,
        "Adam": torch.optim.Adam,
    }

    if name not in optimizers:
        raise f"{name} is not a valid choice of optimizer. Valid choices are {', '.join(list(optimizers.keys()))}"

    flags = {}
    if name == "SGD":
        flags["momentum"] = FLAGS.optimizer_momentum
    print("flags optim", flags)
    optimizer = optimizers[name](params, lr=FLAGS.learning_rate, **flags)
    return optimizer


def onehot2indices(onehot):
    return torch.max(onehot, 1)[1]


def to_numpy(obj):
    if type(obj) != torch.Tensor:
        return obj
    return obj.cpu().detach().numpy()


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
      predictions: 2D float array of size [batch_size, n_classes]
      labels: 2D int array of size [batch_size, n_classes]
              with one-hot encoding. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch

    TODO:
    Implement accuracy computation.
    """
    pred_indices = np.argmax(to_numpy(predictions), axis=1)
    targets_indices = np.argmax(to_numpy(targets), axis=1)
    return np.mean(pred_indices == targets_indices)


def train():
    """
    Performs training and evaluation of MLP model.

    TODO:
    Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
    """

    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    ts = str(int(time.time()))
    print("timestamp:", ts)
    print("device:", DEVICE)

    ## Prepare all functions
    # Get number of units in each hidden layer specified in the string such as 100,100
    if FLAGS.dnn_hidden_units:
        dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
        dnn_hidden_units = [
            int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units
        ]
    else:
        dnn_hidden_units = []

        # neg_slope = FLAGS.neg_slope

    data = cifar10_utils.get_cifar10(FLAGS.data_dir)

    imsize = data["train"].images[0].shape
    n_inputs = imsize[0] * imsize[1] * imsize[2]
    n_classes = data["train"].labels[0].shape[0]

    model = MLP(n_inputs, dnn_hidden_units, n_classes, FLAGS.activation_fn)
    model.to(DEVICE)
    print("model:")
    print(model)

    optimizer = get_optimizer(FLAGS.optimizer, model.parameters())
    scheduler = get_scheduler(FLAGS.scheduler, optimizer)
    loss_fn = F.cross_entropy
    i = 0

    losses = defaultdict(list)
    accs = defaultdict(list)
    i_at_evals = []
    while True:
        model.train()
        optimizer.zero_grad()
        X_train, y_train = data["train"].next_batch(FLAGS.batch_size)
        X_train = torch.from_numpy(X_train).to(DEVICE)
        y_train = torch.from_numpy(y_train).to(DEVICE)

        pred_train = model(X_train)
        loss = loss_fn(pred_train, onehot2indices(y_train))
        loss.backward()

        optimizer.step()
        if scheduler:
            scheduler.step()
            if i % FLAGS.scheduler_step_size == 0 and i > 0:
                print("[scheduler] updating LR")

        epoch = data["train"].epochs_completed
        if i % FLAGS.eval_freq == 0:
            with torch.no_grad():
                model.eval()
                i_at_evals.append(i)
                X_test, y_test = data["test"].images, data["test"].labels
                X_test = torch.from_numpy(X_test).to(DEVICE)
                y_test = torch.from_numpy(y_test).to(DEVICE)
                pred_test = model(X_test)

                train_loss = loss.item()
                test_loss = loss_fn(pred_test, onehot2indices(y_test)).item()
                losses["train"].append(train_loss)
                losses["test"].append(test_loss)

                train_acc = accuracy(pred_train, y_train) * 100
                test_acc = accuracy(pred_test, y_test) * 100
                accs["train"].append(train_acc)
                accs["test"].append(test_acc)
                print(
                    "[epoch %s: %s/%s] train_acc: %0.3f, test_acc: %0.3f"
                    % (
                        epoch,
                        i * FLAGS.batch_size % data["train"].num_examples,
                        data["train"].num_examples,
                        train_acc,
                        test_acc,
                    )
                )

        i += 1
        if i >= FLAGS.max_steps:
            break

    print("highest test accuracy:", np.max(accs["test"]))
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.plot(i_at_evals, losses["train"], label="train loss", color="purple")
    ax.plot(i_at_evals, losses["test"], label="test loss", color="green")
    ax.set_xlabel("batch (step)")
    ax.set_ylabel("loss")
    plt.legend(loc="upper left")
    ax2 = ax.twinx()
    ax2.plot(
        i_at_evals, accs["train"], "-.", color="purple", label="train acc"
    )
    ax2.plot(i_at_evals, accs["test"], "-.", color="green", label="test acc")
    ax2.set_ylabel("accuracy (%)")
    plt.legend(loc="upper right")
    plt.title("loss and accuracy (dashed) when training PyTorch MLP")
    plt.savefig(
        f"{ts}_mlp_pytorch_results.png",
        bbox_inches="tight",
    )


def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + " : " + str(value))


def main():
    """
    Main function
    """
    # Print all Flags to confirm parameter settings
    print_flags()

    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)

    # Run the training operation
    train()


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dnn_hidden_units",
        type=str,
        default=DNN_HIDDEN_UNITS_DEFAULT,
        help="Comma separated list of number of units in each hidden layer",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=LEARNING_RATE_DEFAULT,
        help="Learning rate",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=MAX_STEPS_DEFAULT,
        help="Number of steps to run trainer.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE_DEFAULT,
        help="Batch size to run trainer.",
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=EVAL_FREQ_DEFAULT,
        help="Frequency of evaluation on the test set",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=DATA_DIR_DEFAULT,
        help="Directory for storing input data",
    )

    # my arguments :D
    parser.add_argument(
        "--optimizer",
        type=str,
        default=OPTIMIZER_DEFAULT,
        help="Name of optimizer to use",
    )
    parser.add_argument(
        "--optimizer_momentum",
        type=float,
        default=OPTIMIZER_MOMENTUM_DEFAULT,
        help="momentum of optimizer",
    )
    parser.add_argument(
        "--activation_fn",
        type=str,
        default=ACTIVATION_FN_DEFAULT,
        help="The name of the activation function",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default=SCHEDULER_DEFAULT,
        help="The name of the scheduler",
    )
    parser.add_argument(
        "--scheduler_gamma",
        type=float,
        default=SCHEDULER_GAMMA_DEFAULT,
        help="The value of the scheduler gamma parameter ",
    )
    parser.add_argument(
        "--scheduler_step_size",
        type=int,
        default=SCHEDULER_STEP_SIZE_DEFAULT,
        help="The step size of the scheduler",
    )

    FLAGS, unparsed = parser.parse_known_args()

    main()

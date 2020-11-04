"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
import argparse
import numpy as np
import os
from convnet_pytorch import ConvNet
import cifar10_utils
import time
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.nn as nn

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500

OPTIMIZER_DEFAULT = "ADAM"
OPTIMIZER_MOMENTUM_DEFAULT = 0.9

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = "./cifar10/cifar-10-batches-py"

FLAGS = None


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_optimizer(name, params):
    optimizers = {
        "SGD": torch.optim.SGD,
        "ADAM": torch.optim.Adam,
    }

    if name not in optimizers:
        raise f"{name} is not a valid choice of optimizer. Valid choices are {', '.join(list(optimizers.keys()))}"

    flags = {}
    if name == "SGD":
        flags["momentum"] = FLAGS.optimizer_momentum
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
    Performs training and evaluation of ConvNet model.

    TODO:
    Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
    """

    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    ts = str(int(time.time()))
    print("timestamp:", ts)
    print("device:", DEVICE)

    data = cifar10_utils.get_cifar10(FLAGS.data_dir)

    imsize = data["train"].images[0].shape
    n_channels = imsize[0]
    n_classes = data["train"].labels[0].shape[0]

    model = ConvNet(n_channels, n_classes)
    model.to(DEVICE)
    print("model:")
    print(model)

    optimizer = get_optimizer(FLAGS.optimizer, model.parameters())
    loss_fn = F.cross_entropy
    i = 0

    losses = defaultdict(list)
    accs = defaultdict(list)
    i_at_evals = []
    while True:
        optimizer.zero_grad()
        X_train, y_train = data["train"].next_batch(FLAGS.batch_size)
        X_train = torch.from_numpy(X_train).to(DEVICE)
        y_train = torch.from_numpy(y_train).to(DEVICE)

        pred_train = model(X_train)
        loss = loss_fn(pred_train, onehot2indices(y_train))
        loss.backward()

        optimizer.step()

        epoch = data["train"].epochs_completed
        if i % FLAGS.eval_freq == 0:
            with torch.no_grad():
                model.eval()
                i_at_evals.append(i)
                X_test, y_test = data["test"].images, data["test"].labels
                # test_indices = np.arange(0, X_test.shape[0])
                # np.random.shuffle(test_indices)
                # max_index = int(0.6 * len(test_indices))
                # X_test = X_test[test_indices[:max_index]]
                # y_test = y_test[test_indices[:max_index]]
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

                X_test = None
                y_test = None
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
            model.train()

        i += 1
        if i >= FLAGS.max_steps:
            break

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
    plt.title("loss and accuracy (dashed) when training PyTorch ConvNet")
    plt.savefig(
        f"{ts}_convnet_results.png",
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

    FLAGS, unparsed = parser.parse_known_args()

    main()

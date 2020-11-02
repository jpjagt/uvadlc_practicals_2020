"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils
from collections import defaultdict
import matplotlib.pyplot as plt

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = "100"
LEARNING_RATE_DEFAULT = 1e-3
MAX_STEPS_DEFAULT = 1400
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100
NEG_SLOPE_DEFAULT = 0.02

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = "./cifar10/cifar-10-batches-py"

FLAGS = None


class Optimizer:
    def __init__(self, modules, learning_rate):
        self.modules = modules
        self.learning_rate = learning_rate

    def step(self):
        for module in self.modules:
            if not hasattr(module, "grads"):
                continue

            for key, grads in module.grads.items():
                grads[np.isnan(grads)] = 0.0
                module.params[key] -= self.learning_rate * grads
                module.grads[key] = np.zeros(grads.shape)


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
    pred_indices = np.argmax(predictions, axis=1)
    targets_indices = np.argmax(targets, axis=1)
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

    ## Prepare all functions
    # Get number of units in each hidden layer specified in the string such as 100,100
    if FLAGS.dnn_hidden_units:
        dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
        dnn_hidden_units = [
            int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units
        ]
    else:
        dnn_hidden_units = []

    data = cifar10_utils.get_cifar10(FLAGS.data_dir)

    imsize = data["train"].images[0].shape
    n_inputs = imsize[0] * imsize[1] * imsize[2]
    n_classes = data["train"].labels[0].shape[0]

    mlp = MLP(n_inputs, dnn_hidden_units, n_classes, neg_slope=0)
    optimizer = Optimizer(mlp.modules, FLAGS.learning_rate)
    loss_module = CrossEntropyModule()
    i = 0

    losses = defaultdict(list)
    accs = defaultdict(list)
    i_at_evals = []
    while True:
        X_train, y_train = data["train"].next_batch(FLAGS.batch_size)
        pred_train = mlp.forward(X_train)
        loss = loss_module.forward(pred_train, y_train)

        dout = loss_module.backward(pred_train, y_train)
        mlp.backward(dout)
        optimizer.step()

        epoch = data["train"].epochs_completed
        if i % FLAGS.eval_freq == 0:
            i_at_evals.append(i)
            X_test, y_test = data["test"].images, data["test"].labels
            pred_test = mlp.forward(X_test)

            train_loss = loss
            test_loss = loss_module.forward(pred_test, y_test)
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
    plt.title(
        "loss and accuracy (dashed) when training numpy MLP with default params"
    )
    plt.savefig("train_mlp_numpy_results.png", bbox_inches="tight")


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
    FLAGS, unparsed = parser.parse_known_args()

    main()

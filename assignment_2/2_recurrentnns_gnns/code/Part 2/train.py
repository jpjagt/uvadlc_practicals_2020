# MIT License
#
# Copyright (c) 2019 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2019
# Date Created: 2019-09-06
###############################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from datetime import datetime
import argparse

import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import TextDataset
from model import TextGenerationModel

import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

###############################################################################

losses = defaultdict(list)
accs = defaultdict(list)
timestamp = int(datetime.now().timestamp())


def sample_sents(
    config, model, dataset, temperature=1.0, sample_multiple_lengths=True
):
    seq_len_multiplier = 6
    sent_lens = (
        [
            int(config.seq_length * 0.5),
            config.seq_length,
            int(config.seq_length * seq_len_multiplier),
        ]
        if sample_multiple_lengths
        else [int(config.seq_length * seq_len_multiplier)]
    )
    with torch.no_grad():
        model.eval()
        sample_inputs = (
            torch.arange(0, dataset.vocab_size, step=4)
            .view(-1, 1)
            .to(config.device)
        )
        hidden_and_state = None
        sample_outputs_over_time = [sample_inputs]
        for i in range(int(config.seq_length * seq_len_multiplier)):
            sample_outputs_probs = model(
                sample_inputs,
                hidden_and_state,
            )
            if config.sample_method == "greedy":
                sample_outputs = torch.argmax(sample_outputs_probs, dim=2)
            elif config.sample_method == "random":
                sample_outputs = torch.multinomial(
                    torch.softmax(
                        sample_outputs_probs.squeeze() * temperature,
                        dim=1,
                    ),
                    1,
                )
            else:
                raise ValueError(
                    "please choose a valid sample_method (greedy or random)"
                )
            sample_outputs_over_time.append(sample_outputs)
            sample_inputs = sample_outputs
            hidden_and_state = model.hidden_and_state_out

        batch_sample_outputs = torch.stack(
            sample_outputs_over_time, dim=1
        ).squeeze()
        sample_input_chars = dataset.convert_to_string(
            sample_outputs_over_time[0].squeeze().tolist()
        )
        print("generated sentences:")
        for i, sample_output in enumerate(batch_sample_outputs):
            sent_full = dataset.convert_to_string(sample_output.tolist())
            for sent_len in sent_lens:
                print(
                    f"[len={sent_len}, start={sample_input_chars[i]}] {sent_full[:sent_len]}"
                )
            print()


def plot_result(result, name, ax):
    np_result = np.array(list(result.values()))
    avg_result = np_result.mean(axis=0)
    std_dev = np.std(np_result, axis=0)
    ax.plot(avg_result)
    if len(list(result.values())) > 1:
        ax.fill_between(
            range(np_result.shape[1]),
            avg_result - std_dev,
            avg_result + std_dev,
            alpha=0.5,
        )
    ax.set_ylabel(name)
    ax.set_xlabel("training step")


def plot_results(config):
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    plot_result(losses, "loss", axs[0])
    plot_result(accs, "accuracy", axs[1])
    Path("./plots").mkdir(exist_ok=True)

    book = config.txt_file.split("/")[-1].replace("_", " ").replace(".txt", "")
    plt.suptitle(f"loss and accuracy of LSTM learning from book {book}")
    plt.savefig(f"plots/{timestamp}_gen_results.png", bbox_inches="tight")


def calc_accuracy(outputs, targets):
    outputs_pred = torch.argmax(outputs, axis=2)
    return torch.mean((outputs_pred == targets).float()).item()


def train(config, seed=25):
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file, config.seq_length)
    data_loader = DataLoader(dataset, config.batch_size)

    # Initialize the model that we are going to use
    model = TextGenerationModel(
        config.batch_size,
        config.seq_length,
        dataset.vocab_size,
        lstm_num_hidden=config.lstm_num_hidden,
        lstm_num_layers=config.lstm_num_layers,
        device=config.device,
    ).to(device)

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    # criterion = torch.nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, config.learning_rate_step, gamma=config.learning_rate_decay
    )
    step = 0
    while True:
        for (batch_inputs, batch_targets) in data_loader:
            model.train()
            batch_inputs = torch.stack(batch_inputs, dim=1).to(device)
            batch_targets = torch.stack(batch_targets, dim=1).to(device)

            # Only for time measurement of step through network
            t1 = time.time()

            model.zero_grad()
            batch_outputs = model(batch_inputs)

            loss = criterion(batch_outputs.permute(0, 2, 1), batch_targets)
            loss.backward()

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_norm)

            optimizer.step()
            scheduler.step()

            accuracy = calc_accuracy(batch_outputs, batch_targets)

            # Just for time measurement
            t2 = time.time()
            examples_per_second = config.batch_size / float(t2 - t1)

            losses[str(seed)].append(loss.item())
            accs[str(seed)].append(accuracy)

            if (step + 1) % config.print_every == 0:

                print(
                    "[{}] Train Step {:04d}/{:04d}, Batch Size = {}, \
                        Examples/Sec = {:.2f}, "
                    "Accuracy = {:.2f}, Loss = {:.3f}".format(
                        datetime.now().strftime("%Y-%m-%d %H:%M"),
                        step,
                        config.train_steps,
                        config.batch_size,
                        examples_per_second,
                        accuracy,
                        loss,
                    )
                )

            if (step + 1) % config.sample_every == 0:
                # Generate some sentences by sampling from the model
                if config.sample_method == "random":
                    for temperature in [0.5, 1.0, 2.0]:
                        print()
                        print("**********")
                        print(f"TEMPERATURE: {temperature}")
                        print("**********")
                        print()
                        sample_sents(
                            config,
                            model,
                            dataset,
                            temperature,
                            sample_multiple_lengths=False,
                        )
                else:
                    sample_sents(
                        config,
                        model,
                        dataset,
                        sample_multiple_lengths=True,
                    )

            if step >= config.train_steps:
                break

            step += 1

        if step >= config.train_steps:
            break

    print("Done training.")
    print()
    print()
    print(
        f"max accuracy: {np.max(accs[str(seed)])} at step {np.argmax(accs[str(seed)])}"
    )
    plot_results(config)


###############################################################################
###############################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument(
        "--txt_file",
        type=str,
        required=True,
        help="Path to a .txt file to train on",
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=30,
        help="Length of an input sequence",
    )
    parser.add_argument(
        "--lstm_num_hidden",
        type=int,
        default=128,
        help="Number of hidden units in the LSTM",
    )
    parser.add_argument(
        "--lstm_num_layers",
        type=int,
        default=2,
        help="Number of LSTM layers in the model",
    )

    # Training params
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Number of examples to process in a batch",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-3, help="Learning rate"
    )

    # It is not necessary to implement the following three params,
    # but it may help training.
    parser.add_argument(
        "--learning_rate_decay",
        type=float,
        default=0.96,
        help="Learning rate decay fraction",
    )
    parser.add_argument(
        "--learning_rate_step",
        type=int,
        default=5000,
        help="Learning rate step",
    )
    parser.add_argument(
        "--dropout_keep_prob",
        type=float,
        default=1.0,
        help="Dropout keep probability",
    )

    parser.add_argument(
        "--train_steps",
        type=int,
        default=int(1e6),
        help="Number of training steps",
    )
    parser.add_argument("--max_norm", type=float, default=5.0, help="--")

    # Misc params
    parser.add_argument(
        "--summary_path",
        type=str,
        default="./summaries/",
        help="Output path for summaries",
    )
    parser.add_argument(
        "--print_every",
        type=int,
        default=50,
        help="How often to print training progress",
    )
    parser.add_argument(
        "--sample_every",
        type=int,
        default=100,
        help="How often to sample from the model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=("cpu" if not torch.cuda.is_available() else "cuda"),
        help="Device to run the model on.",
    )

    # If needed/wanted, feel free to add more arguments
    parser.add_argument(
        "--sample_temperature",
        type=float,
        default=1.0,
        help="Temperature to use when sampling",
    )
    parser.add_argument(
        "--sample_method",
        type=str,
        default="greedy",
        help="sample method to use (greedy or random)",
    )

    config = parser.parse_args()

    print("timestamp:", timestamp)
    print(config)

    # Train the model
    train(config)

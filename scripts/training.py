import torch
import torch.onnx
import subprocess
import argparse
import gc
import torch.nn as nn
import torch.functional as F
from fastai.data.all import DataLoader, DataLoaders
from fastai.learner import Learner
from fastai.callback.schedule import lr_find

from model import ResNet
from data import get_datasets

device = torch.device("cuda:0")
input_channels = 7
bootstrap_epochs = 10

parser = argparse.ArgumentParser()
parser.add_argument(
    "--deep_ww", help="Path to deep wallwars executable", default="../build/deep_ww"
)
parser.add_argument("--models", help="Path to store the models", default="../models")
parser.add_argument("--data", help="Path to store training data", default="../data")
parser.add_argument("-c", "--columns", help="Number of columns", default=6, type=int)
parser.add_argument("-r", "--rows", help="Number of rows", default=6, type=int)
parser.add_argument(
    "--generations",
    help="Number of generations to train for",
    default=40,
    type=int,
)
parser.add_argument(
    "--initial_generation",
    help="Initial generation to start from (to continue previous run)",
    default=0,
    type=int,
)
parser.add_argument(
    "--training-batch-size",
    help="Batch size used during training",
    default=512,
    type=int,
)
parser.add_argument(
    "--inference-batch-size",
    help="Batch size used during inference (self-play)",
    default=256,
    type=int,
)
parser.add_argument(
    "--hidden_channels",
    help="Number of channels to use in the hidden layers of the ResNet",
    default=32,
    type=int,
)
parser.add_argument(
    "--layers",
    help="Number of layers in the ResNet",
    default=20,
    type=int,
)
parser.add_argument(
    "--max-training-window",
    help="Determines the maximum number of past generations used for training data",
    default=20,
    type=int,
)
parser.add_argument(
    "--training-games",
    help="Determines the maximum number of games used for training data",
    default=20000,
    type=int,
)
parser.add_argument(
    "--games",
    help="Number of games to play in one iteration of self play",
    default=5000,
    type=int,
)
parser.add_argument(
    "-s",
    "--samples",
    help="Number of samples to use per action during self play",
    default=5000,
    type=int,
)
parser.add_argument(
    "--epochs",
    help="Number of epochs to train per training loop",
    default=1,
    type=int,
)
parser.add_argument(
    "-j",
    "--threads",
    help="Number of threads to use for sample generation during self play",
    default=20,
    type=int,
)
parser.add_argument(
    "--log",
    help="Log file location",
    default="log.txt",
)
args = parser.parse_args()


def get_training_paths(generation):
    lb = max(generation - args.max_training_window, (generation - 1) // 2)
    return [f"{args.data}/generation_{i}" for i in range(lb, generation)]


def save_model(model, name):
    torch.save(model, f"{args.models}/{name}.pt")
    input_names = ["States"]
    output_names = ["Priors", "Values"]
    dummy_input = torch.randn(
        args.inference_batch_size, input_channels, args.columns, args.rows
    ).to(device)
    model.log_output = False
    print("Exporting onnx...")
    torch.onnx.export(
        model,
        dummy_input,
        f"{args.models}/{name}.onnx",
        input_names=input_names,
        output_names=output_names,
    )
    print("Converting onnx to trt...")
    with open(args.log, "a") as f:
        subprocess.run(
            [
                "trtexec",
                f"--onnx={args.models}/{name}.onnx",
                f"--saveEngine={args.models}/{name}.trt",
            ],
            stdout=f,
            stderr=f,
        )
    model.log_output = True


def load_model(name):
    return torch.load(f"{args.models}/{name}.pt").to(device)


def run_self_play(model1, model2, generation):
    print(f"Running self play (generation {generation})...")
    with open(args.log, "a") as f:
        subprocess.run(
            [
                args.deep_ww,
                "-model1",
                model1,
                "-model2",
                model2,
                "-output",
                f"{args.data}/generation_{generation}",
                "-columns",
                str(args.columns),
                "-rows",
                str(args.rows),
                "-j",
                str(args.threads),
                "-games",
                str(args.games),
                "-samples",
                str(args.samples),
            ],
            stdout=f,
            stderr=f,
        )


def predict_valuation(xs):
    return torch.where(xs[1] >= 0.05, 1.0, 0.0) + torch.where(xs[1] <= 0.05, -1.0, 0.0)


def valuation_accuracy(xs, ys):
    return (predict_valuation(xs) == predict_valuation(ys)).float().mean()


def predict_move(xs):
    return torch.max(xs[0], 1).indices


def move_accuracy(xs, ys):
    return (predict_move(xs) == predict_move(ys)).float().mean()


def loss(out, label):
    priors_out, values_out = out
    priors_label, values_label = label

    mse = nn.MSELoss()
    kl_div = nn.KLDivLoss(reduction="batchmean")

    return kl_div(priors_out, priors_label) + mse(values_out, values_label)


def train_model(model, generation, epochs):
    print(f"Loading training data (generation {generation})...")
    training_data, valid_data = get_datasets(
        get_training_paths(generation), args.training_games, args.columns, args.rows
    )
    training_loader = DataLoader(
        training_data,
        bs=args.training_batch_size,
        device=device,
        pin_memory=True,
        shuffle=True,
        num_workers=4,
    )
    valid_loader = DataLoader(
        valid_data,
        bs=args.training_batch_size,
        device=device,
        pin_memory=True,
        num_workers=4,
    )
    loaders = DataLoaders(training_loader, valid_loader)

    learner = Learner(
        loaders, model, loss_func=loss, metrics=[valuation_accuracy, move_accuracy]
    )
    learning_rate = learner.lr_find(show_plot=False)[0]
    print(f"Training generation {generation} with learning rate {learning_rate}...")
    learner.fit(epochs, learning_rate)


# Bootstrap generation 0 data
if args.initial_generation == 0:
    run_self_play("simple", "simple", 0)
    model = ResNet(args.columns, args.rows, args.hidden_channels, args.layers)
    start_generation = 2
    train_model(model, 1, bootstrap_epochs)
    save_model(model, "model_1")
    gc.collect()
else:
    model = load_model(f"model_{args.initial_generation}")
    save_model(model, f"model_{args.initial_generation}")
    start_generation = args.initial_generation + 1


for generation in range(start_generation, start_generation + args.generations - 1):
    run_self_play(f"{args.models}/model_{generation - 1}.trt", "", generation - 1)
    train_model(model, generation, args.epochs)
    save_model(model, f"model_{generation}")
    gc.collect()

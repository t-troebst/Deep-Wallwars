# MCTS with Neural Nets for Wallwars

This project is under active development.

## Overview

The aim of this project is to develop an AlphaGo-inspired MCTS for the abstract strategy game
[Wallwars](https://www.wallwars.net).
In particular, it is supposed to achieve three objectives:

1. Be competitive with humans on larger boards.
2. Beat the negamax-based AI that is already on the website on an 8x8 board.
3. Train a good model in under 24 hours on consumer hardware.

The current status is that a model trained in 3 hours on an RTX 4080 and i7 13700k easily beats both
myself and the negamax AI with milliseconds of thinking time per move.
However, there is still lots of interesting work to be done to create the strongest agent!

## Design

In order to achieve goal 3, the code is written in a way to optimize self-play performance unlike a
lot of toy Python implementations that are out there.

The ResNet model used for inference is compiled into an optimized form using TensorRT and self-play
is done in C++.
Inferences are cached using a sharded LRU cache which saves some work in the case of repeating or
early game positions.
However, the main challenge is that in order to fully utilize a modern GPU, we need to feed a
constant stream of decently sized batches (lets say at least 128 inferences per batch, the bigger
the better) to the GPU.

The idea for implementing this efficiently is that each sample down the tree in our MCTS is a
coroutine.
These coroutines can be scheduled on a thread pool so that multiple samples both from the same MCTS
instance and across different MCTS instances can be in-flight at the same time.
This allows us to efficiently keep 20+ CPU cores 100% used during training even as games are
starting and stopping.

Moreover, when a sample reaches a leaf node where it would need to get an output from our model, it
awaits on a `BatchedModel`.
Effectively, it puts its inference request on a lock-free multi-producer multi-consumer queue and
suspends itself.
One or more worker threads pull inference requests from the queue in batches, have them handled on
the GPU with TensorRT, and then resume the corresponding sample coroutines with their data.
As long as our CPU threads are generating enough samples, this provides a constant stream of equal
sized batches for the GPU.
Ideally at least two worker threads should be used to mitigate the time spent on preparing the input
and delivering the output.

## Workflow

A new model can be trained using `scripts/training.py` (by default it assumes you're executing it
from the `scripts` folder as your working directory).
It has some reasonable defaults and relatively self-explanatory command line arguments.
Fine-tuning the many parameters and the model architecture is work in progress.

## Dependencies (C++)

Required:
* folly (for coroutines, logging, thread pools, and more)
* CUDA & TensorRT (for inference)

Optional:
* Catch2 v3 (for unit tests)

## Dependencies (Python)

* PyTorch (for model creation) w/ onnx (for exporting)
* fastai  (for training)

## GUI Font Setup

The GUI includes a bundled DejaVu Sans font (`assets/gui/fonts/DejaVuSans.ttf`) for reliable text rendering across all platforms. No additional setup is required.

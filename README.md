# MCTS with Neural Nets for Wallwars

This project is in the early stages of development.

## Workflow

We use TensorRT for high-speed inference and PyTorch for training.
As a result, some conversions are necessary and ONNX is used as an intermediate format.

* `scripts/model.py` generates a new model in ONNX format - we fix the batch size for optimal
  performance.
* Once implemented, `scripts/training.py` will also output models in ONNX format.
* Conversion can be done with `trtexec --onnx=model.onnx --saveEngine=model.trt`.
* The C++ code reads the resulting serialized engine (which is not portable).

TODO: eventually we will want int8 quanitization for inference and this needs to be added to the
workflow as well.

## Dependencies (C++)

Required:
* folly (for coroutines, logging, thread pools, and more)
* CUDA & TensorRT (for inference)

Optional:
* Catch2 v3 (for unit tests)

## Dependencies (Python)

* PyTorch (for model creation & training)

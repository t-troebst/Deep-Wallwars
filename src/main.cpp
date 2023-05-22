#include <iostream>

#include <gflags/gflags.h>
#include <folly/logging/xlog.h>

#include "play.hpp"

DEFINE_string(model, "model.rt", "TensorRT model used for self-play");

int main(int argc, char **argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    XLOGF(INFO, "Loading TensorRT Model \"{}\"", FLAGS_model);
}

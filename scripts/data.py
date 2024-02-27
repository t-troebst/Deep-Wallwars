from fastai.data.all import RandomSplitter, get_files
from torch import tensor
from random import sample

input_channels = 7


def tensor_from_csv_line(line):
    return tensor([float(x) for x in line.split(", ")])


def parse_file(file, columns, rows):
    result = []
    with open(file) as f:
        lines = f.readlines()
        for i in range(0, len(lines), 4):
            result.append(
                (
                    tensor_from_csv_line(lines[i]).view(input_channels, columns, rows),
                    tuple(tensor_from_csv_line(lines[i + j]) for j in range(1, 3)),
                )
            )
    return result


def parse_files(files, columns, rows):
    return [entry for file in files for entry in parse_file(file, columns, rows)]


def get_datasets(paths, games, columns, rows, splitter=RandomSplitter()):
    files = [file for path in paths for file in get_files(path)]
    if games < len(files):
        files = sample(files, games)
    training_files, valid_files = splitter(files)
    return parse_files((files[i] for i in training_files), columns, rows), parse_files(
        (files[i] for i in valid_files), columns, rows
    )

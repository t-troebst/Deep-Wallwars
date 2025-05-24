from fastai.data.all import RandomSplitter, get_files
from torch import tensor
from random import sample

def tensor_from_csv_line(line):
    return tensor([float(x) for x in line.split(", ")])


def parse_file(file, input_channels, columns, rows):
    expected_num_values = input_channels * columns * rows
    result = []
    with open(file) as f:
        lines = f.readlines()
        for i in range(0, len(lines), 4):
            parsed_tensor_line = tensor_from_csv_line(lines[i])
            actual_num_values = len(parsed_tensor_line)
            if actual_num_values != expected_num_values:
                print(f"ERROR in file {file}, line index {i}: Data dimension mismatch for board state.")
                print(f"Expected {expected_num_values} values (channels:{input_channels} x cols:{columns} x rows:{rows}).")
                print(f"Found {actual_num_values} values.")
                print("This probably means you are trying to load training data from an older run with different board dimensions.")
                exit(1)
            result.append(
                (
                    parsed_tensor_line.view(input_channels, columns, rows),
                    tuple(tensor_from_csv_line(lines[i + j]) for j in range(1, 3)),
                )
            )
    return result


def parse_files(files, input_channels, columns, rows):
    return [entry for file in files for entry in parse_file(file, input_channels, columns, rows)]


def get_datasets(paths, games, input_channels, columns, rows, splitter=RandomSplitter()):
    files = [file for path in paths for file in get_files(path)]
    if games < len(files):
        files = sample(files, games)
    training_files, valid_files = splitter(files)
    return parse_files((files[i] for i in training_files), input_channels, columns, rows), parse_files(
        (files[i] for i in valid_files), input_channels, columns, rows
    )

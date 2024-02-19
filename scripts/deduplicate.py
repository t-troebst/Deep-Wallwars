import sys
import torch

file = sys.argv[1]

outputs = dict()

i = 0
with open(file) as f:
    lines = f.readlines()
    for i in range(0, len(lines), 5):
        lines[i] = lines[i].strip()
        if lines[i] in outputs:
            output = outputs[lines[i]]
            output[0] += 1
            output[1] += torch.tensor([float(x) for x in lines[i + 1].split(", ")])
            output[2] += torch.tensor([float(x) for x in lines[i + 2].split(", ")])
            output[3] += torch.tensor([float(x) for x in lines[i + 3].split(", ")])
        else:
            outputs[lines[i]] = [
                1,
                torch.tensor([float(x) for x in lines[i + 1].split(", ")]),
                torch.tensor([float(x) for x in lines[i + 2].split(", ")]),
                torch.tensor([float(x) for x in lines[i + 3].split(", ")])
            ]

with open(file, "w") as f:
    for input, output in outputs.items():
        f.write(input + "\n")
        for i in range(1, 4):
            f.write(", ".join(map(str, (output[i] / output[0]).tolist())))
            f.write("\n")
        f.write("\n")


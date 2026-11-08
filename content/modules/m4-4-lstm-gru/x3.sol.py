def params(cell, d, h, torch_style=True):
    gates = {"rnn": 1, "gru": 3, "lstm": 4, "cifg": 3, "peephole": 4}[cell]
    p = gates * (d * h + h * h + (2 if torch_style else 1) * h)
    return p + (3 * h if cell == "peephole" else 0)


def stack_params(cell, d, sizes):
    tot, prev = 0, d
    for h in sizes:
        tot += params(cell, prev, h)
        prev = h
    return tot

from itertools import product


def truth_table_net(f, n):
    ...


def evaluate(net, x):
    ...


xor = lambda a, b: a ^ b
print([evaluate(truth_table_net(xor, 2), x) for x in product((0, 1), repeat=2)])

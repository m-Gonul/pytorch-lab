def explore(threads, init):
    ...


def count_interleavings(lengths):
    ...


inc = [("load", "x", "r"), ("add", "r", 1), ("store", "x", "r")]
print(explore([inc, inc], {"x": 0}))

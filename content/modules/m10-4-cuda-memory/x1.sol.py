def can_run(build, cc):
    best = None
    for kind, ma, mi in build:
        if kind == "sm" and ma == cc[0] and mi <= cc[1]:
            return "cubin"
        if kind == "compute" and (ma, mi) <= tuple(cc):
            best = "jit"
    return best


print(can_run([("sm", 8, 0), ("compute", 8, 0)], (8, 6)))

def inclusive_scan(x, op):
    ...


def exclusive_scan(x, op, identity):
    ...


def kogge_stone(x, op):
    ...


def kogge_stone_inplace(x, order):
    ...


def brent_kung(x, op):
    ...


def hierarchical_scan(x, section, op):
    ...


add = lambda a, b: a + b
print(kogge_stone([3, 1, 7, 0, 4, 1, 6, 3], add), brent_kung([3, 1, 7, 0, 4, 1, 6, 3], add))

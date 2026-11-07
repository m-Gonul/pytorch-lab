def sad(a, b):
    ...


def block_match(prev, cur, bs, R):
    ...


def compensate(prev, mvs, bs):
    ...


prev = [[(x * 7 + y * 3) % 50 for x in range(16)] for y in range(16)]
cur = [row[1:] + row[:1] for row in prev]
print(block_match(prev, cur, 8, 2))

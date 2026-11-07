def sectors(byte_addrs, sector=32):
    ...


def bank_degree(word_addrs, n_banks=32):
    ...


def divergent_paths(conds):
    ...


def occupancy(tpb, regs, smem, max_threads=2048, max_blocks=32, regs_sm=65536, smem_sm=102400):
    ...


print(sectors([4 * t for t in range(32)]), bank_degree([33 * t for t in range(32)]), occupancy(256, 64, 0))

def sectors(byte_addrs, sector=32):
    return len({a // sector for a in byte_addrs})


def bank_degree(word_addrs, n_banks=32):
    banks = {}
    for a in set(word_addrs):
        banks.setdefault(a % n_banks, set()).add(a)
    return max(len(v) for v in banks.values())


def divergent_paths(conds):
    return len(set(conds))


def occupancy(tpb, regs, smem, max_threads=2048, max_blocks=32, regs_sm=65536, smem_sm=102400):
    lim = {"threads": max_threads // tpb, "regs": regs_sm // (regs * tpb), "blocks": max_blocks}
    if smem:
        lim["smem"] = smem_sm // smem
    which = min(lim, key=lambda k: (lim[k], k))
    b = lim[which]
    return b, b * tpb / max_threads, which


print(sectors([4 * t for t in range(32)]), bank_degree([33 * t for t in range(32)]), occupancy(256, 64, 0))

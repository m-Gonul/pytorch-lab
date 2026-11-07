def config_stats(BM, BN, BK, r):
    return {"threads": BM * BN // (r * r), "smem_bytes": 4 * (BM * BK + BK * BN), "regs_acc": r * r,
            "ai_global": 2 * BM * BN * BK / (4 * (BM * BK + BK * BN)), "ai_smem": 2 * r * r / (4 * 2 * r)}


def best_config(candidates, smem_limit=48 * 1024, max_threads=1024, max_acc=64):
    ok = []
    for c in candidates:
        s = config_stats(*c)
        if s["smem_bytes"] <= smem_limit and s["threads"] <= max_threads and s["regs_acc"] <= max_acc and s["threads"] >= 32:
            ok.append((s["ai_global"], s["ai_smem"], c))
    return max(ok)[2] if ok else None


print(config_stats(64, 64, 16, 4))

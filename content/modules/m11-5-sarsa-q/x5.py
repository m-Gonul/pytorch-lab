import numpy as np


def prefix_targets(rewards, ratios, gamma):
    # TODO: finite-horizon mixture of flat returns; Q excludes current-action ratio.
    raise NotImplementedError


def update_episode(Q,C,states,actions,rewards,behavior,gamma):
    # TODO: reverse updates with the current greedy target; keep short prefixes.
    raise NotImplementedError

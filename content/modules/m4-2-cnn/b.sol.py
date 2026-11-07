def conv_out(H, k, s, p):
    return (H + 2 * p - k) // s + 1


def receptive_field(layers):
    r, j = 1, 1
    for k, s in layers:
        r += (k - 1) * j
        j *= s
    return r


def grad_product(local_derivs, residual):
    out = 1.0
    for a in local_derivs:
        out *= (1 + a) if residual else a
    return out


def plan(H, layers):
    sizes = []
    for k, s, p in layers:
        H = conv_out(H, k, s, p)
        sizes.append(H)
    return sizes


print(receptive_field([(3, 1), (3, 2), (3, 1)]))        # 9
print(grad_product([0.1] * 10, False), grad_product([0.1] * 10, True))
print(plan(224, [(7, 2, 3), (3, 2, 1)]))                # [112, 56]

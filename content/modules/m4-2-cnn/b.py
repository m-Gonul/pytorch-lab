def conv_out(H, k, s, p):
    ...


def receptive_field(layers):
    ...


def grad_product(local_derivs, residual):
    ...


def plan(H, layers):
    ...


print(receptive_field([(3, 1), (3, 2), (3, 1)]))        # 9
print(grad_product([0.1] * 10, False), grad_product([0.1] * 10, True))
print(plan(224, [(7, 2, 3), (3, 2, 1)]))                # [112, 56]

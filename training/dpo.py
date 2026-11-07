"""DPO math on summed completion log-probs, independent of the service API.

DPO (Rafailov et al., 2023) uses z = beta * ((logp_c - ref_c) - (logp_r - ref_r)) and the loss -log sigmoid(z).
The model enters only through the summed log-probs of the chosen and rejected completions, so, as in the
AWS cookbook's KTO recipe, the exact gradient can be passed to the service's importance-sampling loss as a
per-sequence advantage: advantage = -dL/dlogp, broadcast over completion tokens.
"""
import math


def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def dpo_term(logp_c, logp_r, ref_c, ref_r, beta):
    """Return (loss, advantage_chosen, advantage_rejected, reward_c, reward_r) for one pair."""
    reward_c, reward_r = beta * (logp_c - ref_c), beta * (logp_r - ref_r)
    z = reward_c - reward_r
    loss = -math.log(sigmoid(z)) if z > -30 else -z
    g = beta * sigmoid(-z)  # dL/dlogp_c = -g, dL/dlogp_r = +g
    return loss, g, -g, reward_c, reward_r


if __name__ == "__main__":
    # Finite-difference check of the gradient against the loss.
    b, c, r, rc, rr, h = 0.1, -200.0, -260.0, -210.0, -250.0, 1e-4
    loss, ac, ar, _, _ = dpo_term(c, r, rc, rr, b)
    dc = (dpo_term(c + h, r, rc, rr, b)[0] - dpo_term(c - h, r, rc, rr, b)[0]) / (2 * h)
    dr = (dpo_term(c, r + h, rc, rr, b)[0] - dpo_term(c, r - h, rc, rr, b)[0]) / (2 * h)
    assert abs(-ac - dc) < 1e-6 and abs(-ar - dr) < 1e-6, (ac, dc, ar, dr)
    assert abs(dpo_term(1, 1, 1, 1, b)[0] - math.log(2)) < 1e-12
    print("dpo math ok", round(loss, 4), round(ac, 4), round(ar, 4))

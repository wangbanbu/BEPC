import math
BOUND_ABS_SLACK=1e-5
BOUND_REL_SLACK=1e-6

def bound_decision(g,tau,gamma,kappa):
    """Exact-rule pruning with conservative outward numerical slack.

    Returns None on the whole boundary interval, including equality. Positive
    slack only requests more full scores; it never relaxes final acceptance.
    """
    if not all(math.isfinite(x) for x in (g,tau,gamma,kappa)):
        return None
    if not 0<=gamma<=1 or kappa<=0:raise ValueError('Invalid BEPC correction bound')
    radius=gamma*kappa
    slack=BOUND_ABS_SLACK+BOUND_REL_SLACK*(abs(g)+abs(tau)+radius)
    if g < tau-radius-slack:return True
    if g > tau+radius+slack:return False
    return None

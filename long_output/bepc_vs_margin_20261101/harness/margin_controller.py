"""Plain native Margin control with exactly the uncapped BEPC tree policy.

T2 installs no visual/text/Draft observers.  The shared speed controller keeps
the same terminal CPU hints, scanner, native-logit reads and KV update code.
The constructor-only response4 value satisfies the frozen parent guard; no
forward or acceptance decision occurs until the live budget is infinity.
"""
from __future__ import annotations

import math

from speed_controller import SpeedController


class NoResponseCapMargin(SpeedController):
    def __init__(self, runtime, model, variant, prompt_ids, image_grid, **kwargs):
        actual = dict(variant)
        if (actual.get('family') != 'T2' or actual.get('no_response_cap') is not True
                or actual.get('response_mismatch_budget') is not None
                or int(actual.get('round_mismatch_budget', 2)) != 2
                or float(actual.get('tau', 1.)) != 1.):
            raise ValueError('Frozen plain Margin requires T2/tau1/round2/no response cap')
        super().__init__(runtime, model, dict(actual, response_mismatch_budget=4),
                         prompt_ids, image_grid, **kwargs)
        self.response_budget = math.inf
        self.variant.update(actual)
        assert self.epcc is None and self.visual is None and self.draft is None

    def final_stats(self):
        value = super().final_stats()
        value.update(no_response_cap=True, response_cap_enabled=False,
            response_mismatch_budget=None, round_mismatch_budget=2,
            all_relaxed_methods_same_budget=True,
            budget_exhaustion_feature_retirement=False,
            retained_score_definition='finite native Target top1-minus-Draft margin <= 1',
            visual_first_prune=False, residual_attenuation=False,
            plain_margin_without_representation_observers=True,
            constructor_bootstrap_response_budget=4,
            constructor_bootstrap_is_init_only=True,
            response_budget_internal_representation='positive infinity; not serialized')
        return value

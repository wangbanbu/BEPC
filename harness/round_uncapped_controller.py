"""Remove only the per-round mismatch cap from unchanged uncapped BEPC.

The old round2 constructor value is initialization-only.  The active budget
becomes infinity before any forward or decision.  Native branch scans still
apply BEPC, EOS, padding, visible boundary and first-longest selection.
"""
from __future__ import annotations
import math

from uncapped_controller import NoResponseCapBEPC


class NoRoundCapBEPC(NoResponseCapBEPC):
    def __init__(self,runtime,model,variant,prompt_ids,image_grid,**kwargs):
        actual=dict(variant)
        if actual.get('no_round_cap') is not True or actual.get('round_mismatch_budget') is not None:
            raise ValueError('Explicit no_round_cap=True and round_mismatch_budget=None required')
        super().__init__(runtime,model,dict(actual,round_mismatch_budget=2),
                         prompt_ids,image_grid,**kwargs)
        self.round_budget=math.inf
        self.variant.update(actual)

    def final_stats(self):
        value=super().final_stats()
        value.update(no_round_cap=True,round_cap_enabled=False,
            round_mismatch_budget=None,response_mismatch_budget=None,
            response_cap_enabled=False,constructor_bootstrap_round_budget=2,
            constructor_bootstrap_is_init_only=True,
            round_budget_internal_representation='positive infinity; not serialized',
            retained_score_definition='unchanged symmetric BEPC gamma=.5; response and round mismatch caps removed')
        return value

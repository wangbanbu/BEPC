"""Response-budget-only BEPC ablation over the isolated frozen controller.

The legacy parent constructor checks round2/response4. Its response4 value is
therefore used only during initialization, before any input is forwarded or any
candidate is evaluated. The active budget is then positive infinity, making
only the response-budget comparisons inactive. Every branch still has the
unchanged two-mismatch limit, and only the selected path spends mismatches.

No Target/Draft forward, score, observer, native margin, tree, KV operation,
EOS rule, visible boundary or branch-selection implementation is replaced.
JSON-facing configuration/statistics use null rather than nonstandard Infinity.
"""
from __future__ import annotations

import math

from bepc_experimental import ExperimentalBEPC


class NoResponseCapBEPC(ExperimentalBEPC):
    """Keep the symmetric gamma=.5 BEPC rule and remove only its response cap."""

    def __init__(self, runtime, model, variant, prompt_ids, image_grid, **kwargs):
        actual = dict(variant)
        if actual.get('no_response_cap') is not True:
            raise ValueError('Explicit no_response_cap=True is required')
        if actual.get('response_mismatch_budget') is not None:
            raise ValueError('Uncapped response_mismatch_budget must be None')
        if int(actual.get('round_mismatch_budget', 2)) != 2:
            raise ValueError('The per-round two-mismatch budget is frozen')
        if any(float(actual.get(key, .5)) != .5 for key in
               ('gamma', 'gamma_rescue', 'gamma_protection')):
            raise ValueError('The original symmetric BEPC gamma=.5 is frozen')
        if (float(actual.get('kappa', 1.)) != 1.
                or float(actual.get('eta', 1e-4)) != 1e-4):
            raise ValueError('Original BEPC kappa=1 and eta=1e-4 are frozen')
        actual.update(response_mismatch_budget=None, round_mismatch_budget=2,
                      gamma=.5, gamma_rescue=.5, gamma_protection=.5,
                      no_response_cap=True)
        bootstrap = dict(actual, response_mismatch_budget=4)
        super().__init__(runtime, model, bootstrap, prompt_ids, image_grid, **kwargs)
        # No request forward/evaluation happens in the inherited constructors.
        # A finite committed count can never trigger the inherited cap/retirement.
        self.response_budget = math.inf
        self.variant.update(actual)

    def final_stats(self):
        result = super().final_stats()
        result.update(
            no_response_cap=True,
            response_cap_enabled=False,
            response_mismatch_budget=None,
            round_mismatch_budget=2,
            all_relaxed_methods_same_budget=False,
            budget_exhaustion_feature_retirement=False,
            constructor_bootstrap_response_budget=4,
            constructor_bootstrap_is_init_only=True,
            response_budget_internal_representation='positive infinity; not serialized',
            retained_score_definition='unchanged symmetric BEPC gamma=.5; response cap removed only',
        )
        return result

"""BEPC generation using the tested controller with both mismatch caps removed."""
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / 'harness'))
from round_uncapped_controller import NoRoundCapBEPC as BEPCController
from compact_logits import tree_decoding
from terminal_update import TerminalUpdate


DEFAULT_VARIANT = {
    'compact_logits': True,
    'terminal_skip': True,
    'controller': 'speed',
    'round_mismatch_budget': None,
    'response_mismatch_budget': None,
    'rich_diagnostics': False,
    'family': 'REFINED_BEPC',
    'rank_u': 8,
    'rank_p': 4,
    'text_window': 32,
    'visual_pool_size': 64,
    'gamma': 0.5,
    'kappa': 1.0,
    'eta': 0.0001,
    'residual_attenuation': False,
    'correction_mode': 'both',
    'tau': 1.0,
    'solver': 'fast',
    'prune': True,
    'visual_first_prune': True,
    'experimental_bepc': True,
    'gamma_rescue': 0.5,
    'gamma_protection': 0.5,
    'no_response_cap': True,
    'no_round_cap': True,
}


def default_variant():
    return dict(DEFAULT_VARIANT)


@torch.inference_mode()
def generate(model, inputs, *, max_new_tokens=512, max_length=2048):
    """Run batch-one, greedy ViSpec generation with uncapped BEPC verification."""
    from vispec.model import spec_model_ours as runtime

    original = {name: getattr(runtime, name) for name in BEPCController.FUNCTIONS}
    controller = None
    try:
        runtime.tree_decoding = tree_decoding
        runtime.update_inference_inputs = TerminalUpdate(
            original['update_inference_inputs'], prompt_length=inputs.input_ids.shape[1],
            max_new_tokens=max_new_tokens, eos_token_id=model.tokenizer.eos_token_id)
        controller = BEPCController(
            runtime, model, default_variant(), inputs.input_ids, inputs.image_grid_thw,
            max_new_tokens=max_new_tokens)
        return model.specgenerate(**inputs, temperature=0.,
                                  max_new_tokens=max_new_tokens, max_length=max_length)
    finally:
        if controller is not None:
            controller.restore()
        for name, function in original.items():
            setattr(runtime, name, function)

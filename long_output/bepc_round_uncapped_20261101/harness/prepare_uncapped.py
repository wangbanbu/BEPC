"""Freeze the single-factor removal of the response-level mismatch budget."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT.parent/'bepc_optimization_20261101'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    assert not path.exists(), path
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def main():
    old = json.loads((PREVIOUS/'configs/frozen_config.json').read_text())
    cfg = copy.deepcopy(old)
    cfg.update(name='BEPC response-budget removal, round2 retained',
        created_utc=datetime.now(timezone.utc).isoformat(), parent_experiment=str(PREVIOUS),
        parent_config_sha256=sha(PREVIOUS/'configs/frozen_config.json'))
    capped = old['methods_by_task']['coco']['bepc_fast']
    uncapped = dict(capped, response_mismatch_budget=None, no_response_cap=True)
    methods = dict(strict_shared=old['methods_by_task']['coco']['strict_shared'],
                   bepc_cap4=capped, bepc_no_response_cap=uncapped)
    cfg['methods_by_task'] = {task: copy.deepcopy(methods) for task in cfg['tasks']}
    cfg['development_methods'] = methods
    names = set(old['harness_sha256']) | {'uncapped_controller.py', 'run_uncapped.py', 'prepare_uncapped.py'}
    cfg['harness_sha256'] = {name: sha(ROOT/'harness'/name) for name in sorted(names)}
    cfg['algorithm_changes'] = dict(
        only_algorithm_change='response_mismatch_budget 4 -> null (no response cap)',
        round_mismatch_budget=2, gamma_rescue=0.5, gamma_protection=0.5,
        unchanged=['tau1', 'kappa1', 'eta1e-4', 'U8', 'P4', 'W32', 'visual_pool64',
                   'residual_attenuationFalse', 'native_tree30_including_root', 'native_KV',
                   'mRoPE', 'causal_tree_attention', 'EOS', '512_token_stop', 'long_prompts'],
        implementation='copied capped engine; uncapped response budget represented as positive infinity only at runtime, explicit JSON null',
        feature_cost='uncapped keeps observer/scoring active after four; no budget-exhausted fast path')
    cfg['statistical_plan'] = dict(bootstrap_replicates=5000, unit='image_id',
        comparisons=['bepc_no_response_cap - bepc_cap4', 'bepc_cap4 - strict_shared',
                     'bepc_no_response_cap - strict_shared'],
        multiplicity='Holm across three planned comparisons within each task and metric',
        timing_repetitions=1, all_results_retained=True, no_parameter_selection=True)
    cfg['noninferiority_plan']['reference'] = 'concurrent bepc_cap4 and strict_shared, separately reported'
    for task, data in cfg['data'].items():
        for split, count in [('dev', 20), ('test', 100)]:
            assert sha(data[split+'_path']) == data[split+'_sha256']
            samples = json.loads(Path(data[split+'_path']).read_text())
            assert len(samples) == count and len({str(s['question_id']) for s in samples}) == count
            for sample in samples:
                assert sha(sample['image_path']) == sample['image_sha256']
    model = json.loads(Path(cfg['model_identity_path']).read_text())
    checked = []
    for item in model['files']:
        stat = Path(item['path']).stat()
        assert stat.st_size == item['size_bytes'] and stat.st_mtime_ns == item['mtime_ns']
        checked.append(dict(path=item['path'], size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                            prior_sha256=item['sha256']))
    save(ROOT/'configs/frozen_config.json', cfg)
    save(ROOT/'configs/design_config.json', cfg)
    save(ROOT/'audit/MODEL_STAT_IDENTITY.json', dict(status='pass', files=checked,
        full_weights_rehashed=False, prior_manifest=cfg['model_identity_path']))
    save(ROOT/'audit/MAIN_READY.json', dict(status='pass', frozen_before_generation=True,
        expected_main_requests=900, expected_development_requests=180,
        config_sha256=sha(ROOT/'configs/frozen_config.json'), previous_code_read_only=True,
        main_requires_development_integrity_check=True))
    save(ROOT/'RUN_STATUS.json', dict(status='prepared',
        updated_utc=datetime.now(timezone.utc).isoformat(), main_expected=900, development_expected=180))
    print('Prepared 180 development and 900 test requests; only response cap removed.')


if __name__ == '__main__':
    main()

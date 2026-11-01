"""Freeze the isolated experiment before any new generation or quality scoring."""
from __future__ import annotations

from datetime import datetime, timezone
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent/'single_pass'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    assert not path.exists(), path
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def main():
    old = json.loads((PARENT/'configs/frozen_config.json').read_text())
    cfg = copy.deepcopy(old)
    cfg['name'] = 'Isolated BEPC engineering and bounded-rescue comparison'
    cfg['created_utc'] = datetime.now(timezone.utc).isoformat()
    cfg['parent_experiment'] = str(PARENT)
    cfg['parent_config_sha256'] = sha(PARENT/'configs/frozen_config.json')
    cfg['parent_task_order'] = old['tasks']
    cfg['tasks'] = ['scienceqa', 'coco', 'textvqa']
    cfg['data'] = {t: old['data'][t] for t in cfg['tasks']}
    cfg['repetitions'] = 1
    cfg['warmups'] = 1
    strict = old['methods_by_task']['coco']['strict_shared']
    original = old['methods_by_task']['coco']['bepc_algorithm']
    fast = dict(original, experimental_bepc=True, gamma_rescue=0.5, gamma_protection=0.5)
    rescue = dict(fast, gamma_rescue=0.625)
    methods = dict(strict_shared=strict, bepc_original=original, bepc_fast=fast, bepc_rescue0625=rescue)
    cfg['methods_by_task'] = {t: copy.deepcopy(methods) for t in cfg['tasks']}
    cfg['development_methods'] = methods
    runtime_names = set(old['harness_sha256']) | {'bepc_experimental.py', 'run_experiment.py', 'prepare_experiment.py'}
    cfg['harness_sha256'] = {name: sha(ROOT/'harness'/name) for name in sorted(runtime_names)}
    cfg['config_finalized_before_generation'] = True
    cfg['algorithm_changes'] = dict(engineering='retire BEPC after response mismatch budget exhaustion; strict-equivalent selection',
        relaxation='only rescue gamma 0.5 -> 0.625; protection gamma remains 0.5',
        selection='single preregistered relaxation, no development/test threshold selection',
        unchanged=['subspaces', 'rank_u8', 'rank_p4', 'text_window32', 'visual_pool64', 'tau1',
                   'kappa1', 'eta1e-4', 'residual_attenuationFalse', 'round_budget2', 'response_budget4',
                   'native_tree', 'KV', 'mRoPE', 'causal_tree_attention', 'EOS', '512token_stop', 'long_prompts'])
    cfg['noninferiority_plan'] = dict(
        scienceqa=dict(metric='accuracy', max_absolute_drop=0.02),
        textvqa=dict(metric='VQA_accuracy', max_absolute_drop=0.02),
        coco=dict(SPICE_max_relative_drop=0.05, CHAIRs_max_absolute_increase=0.02,
                  automatic_only=True, human_factual_quality_status='pending'),
        reference='concurrent strict_shared and bepc_original, reported separately',
        decision='one-sided image-clustered 95% bootstrap bound must satisfy every applicable limit; otherwise not established',
        adequacy='100 test groups and historical reuse limit precision; no claim of guaranteed quality')
    cfg['statistical_plan'] = dict(bootstrap_replicates=5000, unit='image_id',
        comparisons=['bepc_fast - bepc_original', 'bepc_rescue0625 - bepc_fast',
                     'bepc_rescue0625 - strict_shared'],
        predeclared=True, timing_repetitions=1,
        quality_scope='official final answers / automatic full-caption metrics, not human factual ground truth',
        multiplicity='report all planned comparisons; simultaneous success requires all quality constraints; no test-based winner selection')
    identity = json.loads(Path(cfg['model_identity_path']).read_text())
    checked = []
    for item in identity['files']:
        path = Path(item['path'])
        stat = path.stat()
        assert stat.st_size == item['size_bytes'] and stat.st_mtime_ns == item['mtime_ns'], path
        checked.append(dict(path=str(path), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                            prior_sha256=item['sha256']))
    for t, entry in cfg['data'].items():
        for split, count in [('dev', 20), ('test', 100)]:
            assert sha(entry[split+'_path']) == entry[split+'_sha256']
            samples = json.loads(Path(entry[split+'_path']).read_text())
            assert len(samples) == count and len({str(s['question_id']) for s in samples}) == count
            for sample in samples:
                assert sha(sample['image_path']) == sample['image_sha256']
        dev = json.loads(Path(entry['dev_path']).read_text())
        test = json.loads(Path(entry['test_path']).read_text())
        assert not {s['image_sha256'] for s in dev} & {s['image_sha256'] for s in test}, t
    save(ROOT/'configs/frozen_config.json', cfg)
    save(ROOT/'configs/design_config.json', cfg)
    save(ROOT/'audit/MODEL_STAT_IDENTITY.json', dict(status='pass', files=checked,
        prior_manifest=str(cfg['model_identity_path']), full_weights_rehashed=False))
    save(ROOT/'audit/MAIN_READY.json', dict(status='pass', frozen_before_generation=True,
        config_sha256=sha(ROOT/'configs/frozen_config.json'),
        expected_main_requests=1200, expected_development_requests=240,
        prior_sources_copied_byte_identically=True, prior_code_and_results_read_only=True))
    save(ROOT/'RUN_STATUS.json', dict(status='prepared', config_sha256=sha(ROOT/'configs/frozen_config.json'),
        updated_utc=datetime.now(timezone.utc).isoformat()))
    print('Prepared 240 development + 1200 test requests, 3 tasks x 4 arms; old sources unchanged.')


if __name__ == '__main__':
    main()

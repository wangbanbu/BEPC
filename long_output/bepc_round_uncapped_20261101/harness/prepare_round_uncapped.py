"""Freeze a paired two-method ablation, without selecting any new threshold."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT.parent/'bepc_vs_margin_20261101'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    assert not path.exists(), path
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def main():
    old = json.loads((PREVIOUS/'configs/frozen_config.json').read_text())
    cfg = copy.deepcopy(old)
    for key in ('main_method_templates', 'parity_methods', 'parity_pairs', 'selection',
                'selection_sha256', 'secondary_comparisons', 'design_config_sha256',
                'original_run_reuse', 'inherited_parity_requests'):
        cfg.pop(key, None)
    bepc = copy.deepcopy(old['methods_by_task']['coco']['bepc_no_response_cap'])
    no_round_cap = dict(bepc, round_mismatch_budget=None, no_round_cap=True)
    methods = dict(bepc_round2=bepc, bepc_no_mismatch_caps=no_round_cap)
    cfg.update(name='BEPC per-round mismatch cap removal, no response cap in both arms',
        created_utc=datetime.now(timezone.utc).isoformat(), parent_experiment=str(PREVIOUS),
        parent_config_sha256=sha(PREVIOUS/'configs/frozen_config.json'),
        expected_development_requests=120, expected_main_requests=600,
        primary_method='bepc_no_mismatch_caps', reference_method='bepc_round2',
        comparisons=[['bepc_no_mismatch_caps', 'bepc_round2']],
        test_sha256={p.name:sha(p) for p in sorted((ROOT/'tests').glob('*.py'))},
        protocol_sha256=sha(ROOT/'EXPERIMENT_PROTOCOL.md'),
        preparation_sha256=sha(__file__),
        algorithm_changes=dict(comparison='per-round mismatch budget 2 versus no round budget; same BEPC score',
            common_tau=1., round_mismatch_budgets=[2,None], common_response_mismatch_budget=None,
            no_parameter_selection=True, same_engineering=True,
            same_BEPC_observers=True),
        statistical_plan=dict(bootstrap_replicates=5000, unit='physical image/group',
            comparisons=['bepc_no_mismatch_caps - bepc_round2'],
            timing_repetitions=1, no_parameter_selection=True, floating_comparison_tolerance=1e-12,
            multiplicity='one planned method contrast per task/metric; cross-metric exploratory interpretation'),
        user_request='把每轮仍最多接受2个mismatch的约束也取消掉测试一下BEPC的解码速度和生成质量')
    cfg['methods_by_task'] = {t:copy.deepcopy(methods) for t in cfg['tasks']}
    cfg['development_methods'] = copy.deepcopy(methods)
    cfg['noninferiority_plan']['reference'] = 'concurrent BEPC round2, automatic diagnostic only'
    names = set(old['harness_sha256']) | {'round_uncapped_controller.py', 'run_round_uncapped.py', 'prepare_round_uncapped.py'}
    cfg['harness_sha256'] = {name:sha(ROOT/'harness'/name) for name in sorted(names)}
    for task, data in cfg['data'].items():
        for split, count in [('dev',20),('test',100)]:
            assert sha(data[split+'_path']) == data[split+'_sha256']
            assert sha(data[split+'_labels_path']) == data[split+'_labels_sha256']
            samples=json.loads(Path(data[split+'_path']).read_text())
            assert len(samples)==count and len({str(s['group_id']) for s in samples})==count
            for sample in samples:
                assert sha(sample['image_path']) == sample['image_sha256']
    assert sha(cfg['model_identity_path'])==cfg['model_identity_sha256']
    model=json.loads(Path(cfg['model_identity_path']).read_text())
    for item in model['files']:
        stat=Path(item['path']).stat()
        assert stat.st_size==item['size_bytes'] and stat.st_mtime_ns==item['mtime_ns']
    for item in json.loads((ROOT/'audit/PRIOR_PRESERVATION.json').read_text())['files']:
        assert sha(item['path'])==item['sha256'],item['path']
    shutil.copyfile(PREVIOUS/'audit/CLEAN_CHECKOUT.json', ROOT/'audit/CLEAN_CHECKOUT.json')
    save(ROOT/'configs/frozen_config.json',cfg)
    save(ROOT/'configs/design_config.json',cfg)
    save(ROOT/'audit/MAIN_READY.json',dict(status='pass',config_sha256=sha(ROOT/'configs/frozen_config.json'),
        expected_main_requests=600,expected_development_requests=120,
        main_requires_development_integrity_check=True, no_threshold_selection=True))
    save(ROOT/'RUN_STATUS.json',dict(status='prepared',main_expected=600,development_expected=120,
        updated_utc=datetime.now(timezone.utc).isoformat()))
    print('Frozen: 120 development + 600 main requests, same BEPC, round2 versus no round cap, no threshold selection',flush=True)


if __name__=='__main__':
    main()

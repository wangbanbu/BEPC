"""Merge audited interpretation into the complete report without changing data."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    report = ROOT / 'reports/FULL_EXPERIMENT_REPORT.md'
    interpretation = ROOT / 'reports/RESULT_INTERPRETATION.md'
    seal = ROOT / 'audit/FULL_REPORT_DELIVERY.json'
    record_path = ROOT / 'audit/COMPLETE_REPORT_INTERPRETATION_MERGE.json'
    assert not record_path.exists()
    previous = json.loads(seal.read_text())
    assert previous['status'] == 'completed'
    for relative, digest in previous['files'].items():
        assert sha(ROOT / relative) == digest, relative
    for name in ('FINAL_DELIVERY_VALIDATION.json', 'SPICE_RESOURCE_RECOVERY_VALIDATION.json'):
        assert json.loads((ROOT / 'audit' / name).read_text())['status'] == 'pass'
    additions = [interpretation, ROOT / 'audit/GQA_FORMAT_DIAGNOSTIC.json',
                 ROOT / 'audit/GQA_FORMAT_DIAGNOSTIC.md']
    assert all(path.is_file() for path in additions)
    original = report.read_text(encoding='utf-8')
    marker = '<!-- verified-interpretation-merged -->'
    assert marker not in original
    title, body = original.split('\n', 1)
    text = interpretation.read_text(encoding='utf-8')
    assert text.startswith('# ')
    text = text.split('\n', 1)[1].lstrip('\n')
    merged = title + '\n\n' + marker + '\n\n## 结果解读与可用性边界\n\n' + text + '\n\n' + body.lstrip('\n')
    evidence = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        purpose='Integrate the evidence-based interpretation, GQA formatting limitation and negative results',
        source_sha256=sha(__file__),
        previous_report_sha256=sha(report),
        previous_delivery_manifest_sha256=sha(seal),
        previous_delivery_manifest=previous,
        added_document_sha256={str(path.relative_to(ROOT)): sha(path) for path in additions},
        primary_tables_metrics_configuration_and_inference_unchanged=True,
        original_report_body_retained=True,
    )
    report.write_text(merged, encoding='utf-8')
    evidence['merged_report_sha256'] = sha(report)
    record_path.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    updated = dict(previous)
    updated['files'] = dict(previous['files'])
    updated['files']['reports/FULL_EXPERIMENT_REPORT.md'] = sha(report)
    for path in additions + [record_path]:
        updated['files'][str(path.relative_to(ROOT))] = sha(path)
    updated['interpretation_postprocessing'] = dict(
        audit_path=str(record_path.relative_to(ROOT)), audit_sha256=sha(record_path),
        numerical_results_changed=False, final_report_sha256=sha(report),
    )
    seal.write_text(json.dumps(updated, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(report)


if __name__ == '__main__':
    main()

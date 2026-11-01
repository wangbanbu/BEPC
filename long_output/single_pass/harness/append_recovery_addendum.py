"""Add the actual official-scoring resource recovery to the complete report.

Preserve the compiler delivery record inside the extended delivery manifest;
the report's numerical tables and all statistical/source seals remain unchanged.
"""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    recovery = json.loads((ROOT / 'audit/SPICE_RESOURCE_RECOVERY.json').read_text())
    validation = json.loads((ROOT / 'audit/SPICE_RESOURCE_RECOVERY_VALIDATION.json').read_text())
    assert recovery['status'] == validation['status'] == 'pass'
    report = ROOT / 'reports/FULL_EXPERIMENT_REPORT.md'
    seal = ROOT / 'audit/FULL_REPORT_DELIVERY.json'
    previous = json.loads(seal.read_text())
    original_seal_sha = sha(seal)
    original_report_sha = sha(report)
    assert previous['files']['reports/FULL_EXPERIMENT_REPORT.md'] == original_report_sha
    assert 'postprocessing' not in previous
    appendix = '''
## 官方 SPICE 资源故障与恢复

首次正式评分中，LVSpec paper 的官方 SPICE/Stanford 解析器在 16GiB Java 堆上限下出现 OutOfMemoryError，跳过句子并最终退出失败；该次尝试没有产生有效的 SPICE 分数，也没有进入最终比较。原失败日志、输入和缓存已完整归档。

恢复仅将这一个官方评分进程的 Java 堆上限提高到 64GiB，保留相同 JAR、模型、线程参数、100条完整输入及参考答案，并使用全新缓存。没有重新生成 Caption，没有改变模型、阈值、样本、指标规则或任何解码计时。其他已成功评分直接复用。

恢复输出须满足100个图像ID齐全且无重复、无 OOM/忽略解析/批评分失败；输入字节、4200条推理记录、所有冻结源码配置和既有成功评分的 SHA 均独立核对。该资源调整属于处理执行故障，不是依据测试质量更改评价标准。LMDB 长字符串缓存写入警告与忽略解析分别审计，不能将两者混为一谈。

冻结评分包装器在读取相同输入的已验证结果时，仍在 provenance 中记录其计划的16GiB命令；此次结果实际由64GiB进程生成，以独立恢复审计中的真实命令和输出SHA为准。

证据：[资源恢复记录](../audit/SPICE_RESOURCE_RECOVERY.json)、[独立恢复验证](../audit/SPICE_RESOURCE_RECOVERY_VALIDATION.md)、[原失败流水线状态](../audit/SPICE_RESOURCE_PIPELINE_FAILED_STATUS.json)。
'''
    extra = ROOT / 'reports/SPICE_RESOURCE_RECOVERY.md'
    assert not extra.exists()
    extra.write_text('# 官方 SPICE 资源恢复说明\n' + appendix.split('\n', 2)[2], encoding='utf-8')
    report.write_text(report.read_text(encoding='utf-8') + '\n' + appendix, encoding='utf-8')
    extended = dict(previous)
    extended['files'] = dict(previous['files'])
    extended['files']['reports/FULL_EXPERIMENT_REPORT.md'] = sha(report)
    for path in (extra, ROOT / 'audit/SPICE_RESOURCE_RECOVERY.json',
                 ROOT / 'audit/SPICE_RESOURCE_RECOVERY_VALIDATION.json'):
        extended['files'][str(path.relative_to(ROOT))] = sha(path)
    extended['postprocessing'] = dict(
        purpose='Disclose the failed official-scoring attempt and audited resource-only recovery',
        source_sha256=sha(__file__),
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        original_compiler_report_sha256=original_report_sha,
        original_compiler_manifest_sha256=original_seal_sha,
        original_compiler_manifest=previous,
        numerical_tables_and_primary_results_unchanged=True,
    )
    seal.write_text(json.dumps(extended, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(report)


if __name__ == '__main__':
    main()

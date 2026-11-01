"""Integrate all sealed one-pass results and the independent audit in one report."""
from pathlib import Path
import csv,hashlib,json
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def number(row,key):
 value=row.get(key,'');return None if value=='' else float(value)
def fmt(x,digits=2):return 'N/A' if x is None else f'{x:.{digits}f}'
def scaled(x,factor):return None if x is None else x*factor
def ratio(a,b):return None if a is None or b in (None,0) else a/b
def quality(task,row):
 if task=='mme':return 'MME '+fmt(number(row,'MME_subset_score'))+' / Acc '+fmt(scaled(number(row,'accuracy'),100))+'%'
 if task=='coco':return 'SPICE '+fmt(scaled(number(row,'SPICE'),100))+' / CHAIRs '+fmt(scaled(number(row,'CHAIRs'),100))+'%'
 return fmt(scaled(number(row,'accuracy'),100))+'%'
def main():
 cfg=json.loads((ROOT/'configs/frozen_config.json').read_text())
 audit=json.loads((ROOT/'audit/FINAL_DELIVERY_VALIDATION.json').read_text())
 assert cfg['repetitions']==1 and audit['status']=='pass'
 rows=list(csv.DictReader((ROOT/'reports/BENCHMARK_SUMMARY.csv').open(encoding='utf-8')))
 by={(row['task'],row['method']):row for row in rows};assert len(rows)==35
 stamp=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
 lines=['# 五任务长输出单次解码性能：完整实验报告','',f'报告生成时间：{stamp}（Asia/Shanghai）。','',
 '## 完成范围与证据','',
 '按用户最新要求，正式测试每题/方法仅生成一次。五个数据集各100组测试，MME为完整双问题组，共600题；七个比较臂合计4200条正式请求。开发集九配置×120题=1080条，其中1003条复用已完成的旧repeat0，只新增77条补齐。原repeat1/2保留在旧实验中，不进入本轮主结果，实际已花费的计算不隐去。',
 '所有图像、题目、官方答案、长提示、模型、512-token预算、原树、T=0及既有接受规则不变。统计仅按固定repeat0；置信区间来自5000次图像/组配对重采样，不包含重复计时波动，也不能证明跨运行稳定性。旧核心的120条parity/48组对照作为继承证据，本轮没有额外重跑。','',
 '## 核心结果总览','',
 '下表给点估计；完整七方法结果、95%CI、原生MAT、有效MAT、显存、解析失败及截断均见下方详细报告。token/s分母包含controller、视觉编码、prefill和全部解码/在线特征；分子不含EOS。MAT仅计接受的Draft，不含Target root；AR不适用。','',
 '| 任务 | 方法 | token/s | 有效Draft MAT | 请求ms | 官方任务质量 |',
 '|---|---|---:|---:|---:|---|']
 for task in cfg['tasks']:
  for method in ['ar','vispec','strict_shared','bepc_algorithm','margin4_ccve','lvspec_paper','lvspec_adapted']:
   row=by[task,method]
   lines.append('| '+' | '.join([task,method,fmt(number(row,'throughput_tokens_per_second')),
    fmt(number(row,'effective_mean_accepted_draft_length')),fmt(scaled(number(row,'mean_seconds'),1000)),quality(task,row)])+' |')
 lines+=['','## 接受策略收益与工程收益','',
 '比较算法时优先参考共享Strict：原生ViSpec额外保留完整leaf logits和最后未使用Draft构树，宽松方法与共享Strict均享有共同工程优化。下面比值只作描述性点估计；统计检验及Holm结论以配对CSV为准，不据MAT或token/s单项宣布总体质量—速度优胜。','',
 '| 任务 | BEPC token/s / 共享Strict | CCVE token/s / 共享Strict | BEPC请求速度比 | CCVE请求速度比 |',
 '|---|---:|---:|---:|---:|']
 for task in cfg['tasks']:
  s=by[task,'strict_shared'];b=by[task,'bepc_algorithm'];c=by[task,'margin4_ccve']
  lines.append('| '+' | '.join([task,fmt(ratio(number(b,'throughput_tokens_per_second'),number(s,'throughput_tokens_per_second')),4),
   fmt(ratio(number(c,'throughput_tokens_per_second'),number(s,'throughput_tokens_per_second')),4),
   fmt(ratio(number(s,'mean_seconds'),number(b,'mean_seconds')),4),fmt(ratio(number(s,'mean_seconds'),number(c,'mean_seconds')),4)])+' |')
 comparisons=[]
 for task in cfg['tasks']:
  lookup={}
  for path in (ROOT/'results/main').glob('main_'+task+'_shard*.jsonl'):
   for line in path.open(encoding='utf-8'):
    row=json.loads(line);lookup[row['method'],str(row['question_id'])]=row
  questions=sorted({q for m,q in lookup if m=='ar'})
  changed=[q for q in questions if lookup['ar',q]['returned_token_ids']!=lookup['vispec',q]['returned_token_ids']]
  shared=[q for q in questions if lookup['strict_shared',q]['returned_token_ids']!=lookup['vispec',q]['returned_token_ids']]
  comparisons.append(dict(task=task,questions=len(questions),AR_native_token_differences=len(changed),native_shared_token_differences=len(shared),different_question_ids=changed))
 comparison_path=ROOT/'audit/AR_STRICT_OUTPUT_COMPARISON.json';assert not comparison_path.exists()
 comparison_path.write_text(json.dumps(dict(causes_not_established=True,no_new_inference=True,tasks=comparisons),indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
 lines+=['','本轮AR/native输出对照（仅重读已保存token，未增加模型调用；成因未定位）：','',
  '| 任务 | 题数 | AR/native token差异 | native/shared token差异 |','|---|---:|---:|---:|']
 for item in comparisons:lines.append(f'| {item["task"]} | {item["questions"]} | {item["AR_native_token_differences"]} | {item["native_shared_token_differences"]} |')
 lines+=['','## 解释边界与审计','',
 '- 本轮为单次测量，速度排名需结合图像抽样CI及输出长度；高token/s可能伴随更长文本、更多错误或更慢请求。没有再执行三次主测。',
 '- QA仅按唯一终止Final answer行评估任务答案，格式/截断失败计0并留在分母；解释文本的事实与推理正确性没有人工Ground Truth。COCO完整长描述用官方SPICE、CHAIR及CIDEr-D；CIDEr长度惩罚可能造成数值极小，不将其单独当事实伤害。',
 '- LVSpec论文比例及开发适配均完整保留。适配若关闭宽松，明确是Strict fallback；不能将其表现称为成功宽松策略。开发集选参不会使用正式测试结果。',
 '- AR显存包含常驻未使用Draft，不是Target-only部署footprint；reserved受allocator历史影响。生成核心、Target/Draft调用、停止、位置/KV和来源映射由独立审计核对。AR/native在前轮存在未定位的输出差异，不能从native/shared一致推出AR/native逐token等价。',
 '- 原始图像开发/测试独立，但MME及已声明的其他历史曝光仍在；全项目未见图像泛化主张No-Go。100组/任务与自动指标不足以证明总体非劣效或人类事实安全。',
 '- 初次单次预检使用base Python/Pillow时出现decoded RGB环境差异，原No-Go记录保留；指定spec Python环境复检通过，原图字节没有更换。正式推理、评测和最终审计使用指定环境。','',
 '## 数据与附件','',
 '- [完整结果CSV](BENCHMARK_SUMMARY.csv) · [全部预设配对统计](PAIRED_COMPARISONS.csv)',
 '- [吞吐/时延/MAT图](BENCHMARK_PERFORMANCE.pdf) · [官方质量图](BENCHMARK_QUALITY.pdf)',
 '- [独立最终审计](../audit/FINAL_DELIVERY_VALIDATION.md) · [原始运行库存与复用](../audit/ORIGINAL_RUN_INVENTORY.md)',
 '- [单次冻结配置](../configs/frozen_config.json) · [统计/输出SHA封印](../audit/STATISTICAL_ANALYSIS.json)',
 '- [实验协议](../EXPERIMENT_PROTOCOL.md) · [单次质量协议](QUALITY_PROTOCOL.md)','',
 '## 逐任务全部结果与统计说明','']
 original=(ROOT/'reports/BENCHMARK_REPORT.md').read_text(encoding='utf-8')
 lines.append(original.split('\n',1)[1] if original.startswith('# ') else original)
 target=ROOT/'reports/FULL_EXPERIMENT_REPORT.md';assert not target.exists();target.write_text('\n'.join(lines),encoding='utf-8')
 paths=[target,ROOT/'reports/BENCHMARK_REPORT.md',ROOT/'audit/FINAL_DELIVERY_VALIDATION.json',ROOT/'audit/STATISTICAL_ANALYSIS.json',ROOT/'audit/ORIGINAL_RUN_INVENTORY.json',comparison_path]
 seal=ROOT/'audit/FULL_REPORT_DELIVERY.json';assert not seal.exists()
 seal.write_text(json.dumps(dict(status='completed',repetitions=1,main_requests=4200,created_utc=datetime.now(timezone.utc).isoformat(),
  files={str(p.relative_to(ROOT)):sha(p) for p in paths},original_primary_results_unchanged=True),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(target)
if __name__=='__main__':main()

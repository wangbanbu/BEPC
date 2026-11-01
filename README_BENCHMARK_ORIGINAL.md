# 干净 ViSpec：五任务实际解码比较

完整交付：[FULL_EXPERIMENT_REPORT.md](reports/FULL_EXPERIMENT_REPORT.md)；[结果CSV](reports/BENCHMARK_SUMMARY.csv)；[最终独立审计](audit/FINAL_DELIVERY_VALIDATION.md)。评分、统计和报告于2026-11-01补完，未重新生成答案。

本目录是 2026-10-31 新建的独立复现实验。官方 ViSpec checkout 保持原样；方法移植、评测和运行管理均在 `harness/`。旧项目未修改。

- 用户固定规模：MME、GQA、TextVQA、ScienceQA 图像题、COCO Caption，各20组开发＋100组测试，所有方法重复3次；MME保留每组完整双问题。
- 主比较：AR、原生ViSpec、共享工程优化Strict、BEPC算法变体、Margin4+CCVE、LVSpec论文比例、LVSpec开发适配比例。
- T=0、原ViSpec树；固定短答案/单句Caption提示和128-token预算。详见 [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md)。
- LVSpec阈值仅由独立开发集按预定规则选择，测试集不调参。冻结规则和实际选择分别在 `configs/design_config.json`、`configs/lvspec_selection.json`；主测使用 `configs/frozen_config.json`。
- 历史曝光勘误不可忽略：本轮开发/测试图像完全分离，但MME全部历史复用，另有7组非MME历史曝光；有效计数以 `audit/DATA_HISTORICAL_OVERLAP_SUPPLEMENT.json` 为准，不能使用冻结输入中原始 `fresh_to_project` 字段声称全新未见泛化。

## 结果与核验

最终正式表格、配对按图像bootstrap置信区间、图和结论由 `harness/analyze_results.py` 写入 `reports/`；逐请求原始记录在 `results/`，官方评分在 `metrics/`。报告生成前不得把开发结果当作测试成绩。

主要审计入口：

- `audit/CLEAN_CHECKOUT.json`：新clone身份、原始tracked文件摘要。
- `audit/ADAPTER_SOURCE_PROVENANCE.json`：BEPC/CCVE移植来源。
- `audit/PARITY_GATE.json`：真实GPU等价检查。
- `audit/METRIC_PARSER_FREEZE.json`：首次正式评分前固定的ScienceQA解析器。
- `reports/LVSPEC_IMPLEMENTATION_AUDIT.md`：论文规则、官方代码差别与ViSpec树适配。
- `reports/METRIC_AUDIT.md`：官方评测来源、格式与局限。
- `audit/FINAL_DELIVERY_VALIDATION.json`：全部主测和分析结束后执行的独立全量审计；文件出现且通过前，不能声称最终交付已通过。

## 运行入口

Python：`/data/wbb/miniconda3/envs/spec/bin/python`。GPU推理使用两张L20，每个图像组固定到一张卡；方法顺序随机化。以下为阶段入口，已有成功结果不得为优化结论重跑或覆盖：

```bash
# 两个shard分别在CUDA_VISIBLE_DEVICES=0、1执行
python harness/run_benchmark.py --phase development --shard 0
python harness/select_lvspec.py
python harness/prepare_experiment.py --stage freeze
python harness/run_benchmark.py --phase main --shard 0
# 主测两个shard、五任务全部完成后执行分析和独立审计
python harness/analyze_results.py --stage all
python harness/validate_delivery.py --phase all
```

每阶段配置、输入、源码、输出与评分摘要均以SHA追溯。计时以CUDA同步后的完整模型请求为主，预处理另记；AR显存包含空闲常驻Draft模型。100组/任务属于小样本比较，COCO自动指标不能替代人工事实质量审核。

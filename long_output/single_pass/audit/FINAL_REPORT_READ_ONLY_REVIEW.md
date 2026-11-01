# 最终报告独立只读复核

复核对象：`long_output/single_pass`。模式：数字、结果到结论及交付完整性审计。复核日期：2026-11-01（Asia/Shanghai）。

结论：**交付复核 PASS，无阻塞性交付问题。** 此结论表示报告忠实呈现已有实验，不能代替质量非劣效、人工事实安全或 AR/树形解码等价性认证。

本次只重读已保存的文件并做确定性算术、SHA 和链接检查；未执行 GPU 推理、Java 评分、阈值选择或新统计检验，未改动既有源码、配置、原始记录、评分、统计表及主报告。唯一新增文件为本复核记录。

## 独立核查结果

| 核查 | 结果与证据 |
|---|---|
| 正式测试规模 | 10 个主 JSONL/完成标记，共 4,200 条；7 方法各 600 题，MME 各 200 题，其他任务各 100 题；全部 `repeat=0`，未混入 repeat1/2。 |
| 主表网格 | `BENCHMARK_SUMMARY.csv` 35 行、35 个唯一 task/method 组合；每行 `repetitions=1`、`quality_repeat=0`。 |
| 时间与吞吐 | 逐个组合由原始记录重算 `sum(content tokens)/sum(request seconds)`、decode-only 吞吐、平均请求时间、平均可见长度，与 CSV 一致。报告区分模型请求耗时和复用的处理器/PIL/H2D 预处理。 |
| MAT 与显存 | 逐个组合重算原生/有效 Draft 接受总数除以对应轮数，以及最大 allocated 峰值，与 CSV 一致；MAT 不含 root，AR 为不适用。 |
| 官方质量点估计 | CSV 中全部适用的 Accuracy、MME、SPICE、CHAIRi、CHAIRs、CIDEr 点估计与已封印的逐方法 aggregate 一致；未发现 NaN/Inf 字符串。 |
| 统计解释 | BEPC/shared 的五任务吞吐、有效 MAT 正向差值与 Holm p<0.01，以及五任务请求时延 CI 全跨 0，均与配对 CSV 一致。COCO BEPC/shared 的 SPICE、CHAIRs、CHAIRi 区间均跨 0。报告未将不显著解释为非劣效。 |
| 原始数据与源码 | 97 项 SHA 复核全部匹配：10 个主结果、5 个统计输出、17 个运行源码、30 个冻结数据/标签文件和 35 个评分 seal。 |
| 主独立审计 | `FINAL_DELIVERY_VALIDATION.json` 为 pass；其“全项目未见图像”主张的 No-Go 被保留，没有误写为全新图像成绩。 |
| 资源恢复审计 | `SPICE_RESOURCE_RECOVERY_VALIDATION.json` 为 pass；16G OOM 失败归档与 64G 同输入官方评分恢复已披露，不当作重复 GPU 计时。 |
| 最新交付 seal | `FULL_REPORT_DELIVERY.json` 的 13 项文件 SHA 全部匹配；解释合并记录保留前一 manifest 和原报告 SHA，统计表/主结果未变。 |
| 文件与链接 | 报告目录 Markdown 中 46 个本地文件链接均存在；性能 PDF 有效 3 页，质量 PDF 存在且标注自动任务指标及 SPICE 百分制。 |

最终复核的主报告 SHA256：`72fee7e1835377f864bbd01aa3852b60cd9a82c51f62185a5e2421c98e619da4`。

## 重要结论边界已进入完整报告

- **GQA 格式地板**：重新从逐题评分按仅 `casefold()` 的既定诊断重算，所有诊断计数与记录一致。Strict 测试原正确 2 题，另 54 题仅大小写不符；开发 Strict 和四个 LV 比例均为 0，故 `.7` 开发不下降门没有判别力。报告开头直接披露，未把诊断改成新主成绩、重选阈值或能力排名。
- **LVSpec fallback**：MME、TextVQA、ScienceQA、COCO 的适配臂显式退回 Strict；GQA 适配与 paper 同为 `.7`。这些情况均已披露，未将 fallback 当作宽松算法收益。
- **触顶范围**：AR/native/shared/BEPC/CCVE 均未触顶，四任务 fallback 未触顶；GQA adapted 与 paper 均有 19% 触顶。已核对报告修正后的范围，未沿用“所有非 paper 均未触顶”的过宽表述。
- **AR 对齐**：600 对中 AR/native 有 427 对可见 token 不同，native/shared 600 对均相同；报告保留成因未定位，未直接归因于 BF16 或缓存，也未以停止观测不同宣称停止规则不同。
- **单次计时与质量**：5000 次图像/组 bootstrap 不代表重复运行稳定性；自动 COCO 指标不构成人工事实 Ground Truth。更高 token/s/MAT 与更短整题时延及保质分别呈现，负结果保留。

## 未解决但不应阻止如实交付的研究问题

GQA 输出格式与官方 exact-match 契约需要在下一轮开发前修正并冻结；AR/native 首分叉需要定位；COCO 需要独立双人图像条件盲审。现有单次固定子集及自动指标仍不足以支持“整体更快且保质”“无损 AR 等价”或“人类事实伤害更少”。这些是后续科学结论的门槛，不是通过修改本轮结果可以消除的缺陷。

本复核独立核对点估计和封印，不重新实现所有 bootstrap 区间、Stanford 场景图解析或官方评价器；上述步骤的执行及资源恢复沿用已保存的独立审计证据。未发现报告内需要更改原始数据才能修复的数字冲突或选择性汇报。

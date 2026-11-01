# 独立评分与统计审查

结论：**PASS with stated limits**。开发阶段已完成，主测试仍在运行；本审查只读取代码、已有数据及评分，不运行 GPU/Java/官方指标，不修改冻结源码。Parser 与统计 self-test 通过。6 个 metric freeze 源码 SHA 全匹配，已生成开发评分的 aggregate/per-question/provenance 封印匹配。

## 已核对

1. **官方评分与解析器**：导入上轮 SHA 冻结的原评测适配器；ScienceQA 固定末行提取后用官方多选题分数，TextVQA 固定末行提取后用官方 VQA-style 分数；失败保留并计 0。COCO 整段文字原样输入，CIDEr-D/SPICE/作者 CHAIR 实现复用。
2. **完整语料复用**：reuse signature 包含 task、完整有序带标签 samples、原始输出全文、固定 parser version 和 official evaluator SHA；再次检查逐题顺序 ID/全文相等。复用完整语料结果，不拼接单句 CIDEr，因而不会改变参考文档频率 corpus。Strict/original 可来自历史、fast 可来自同轮 original，时间始终来自新运行。
3. **SPICE64**：classpath、scorer class、subset/silent/threads/ActiveProcessorCount 与旧调用一致，仅 heap 16G→64G；fresh cache/output 位于新目录。Java 非零返回、OOM/NPE、ignore-sentence/batch failure 阻止评分封印。LMDB cache 持久化警告与丢句分开。实际依赖 JAR SHA 记录在新 SPICE provenance。
4. **统计**：三个任务的 dev/test 均为每题一个独立 physical image/group，检查到 20/100 个唯一 image SHA，因此 group_id bootstrap 与按图像聚类一致。每任务 5000 次共同配对抽样；token/s 为 token 总量/请求总时长，MAT 为接受 token 总量/round 总量。CHAIRi 复采样分子/分母总量，避免逐句均值混淆。点估计与官方 aggregate 再次核对。
5. **非劣效检查**：QA 单侧 95% 差值下界不低于 -0.02；COCO SPICE 检查 new-.95*reference 下界>=0，CHAIRs 检查 new-reference 上界<=0.02，二门同时满足。报告称自动 proxy diagnostic；没有以 p>.05 推断相等。新 gamma .625 在新质量产生前冻结，未按质量挑参。
6. **比较计划**：原配置的三个主比较与新增 original-Strict contextual 比较已有 pre-quality clarification；Holm 按每 task/metric 四组一起修正，较只计三组更保守，但不是跨任务/指标全局控制。
7. **完整性**：阶段 loader 核对完整 method×ordinal×repeat0 网格、config/source/marker SHA、相同输入与合法 shard；GPU runtime 逐题比较 original/fast raw/visible tokens 及接受路径。开发独立验证 240 条已通过。主测试与所有旧文件最终仍需封存后复核。

## 最终报告必须保留的限制

- CI 对固定官方 per-image scores 复采样。尤其 CIDEr 的参考 document frequencies 固定为本轮完整 dev20/test100 corpus，不在每次 bootstrap 后重建 DF；因此是该参考语料下的条件 CI，不能称完全重评语料 CI。
- 零分歧时非参数 bootstrap 差值区间可能退化为 0；自动门 PASS 不证明未见图像上的罕见退化率为 0，更不证明总体非劣效。工程版有额外决策等价实现依据，放宽版没有相同保证。
- 100 图与一次计时不足以证明跨运行速度稳定；paired bootstrap 是图像组成不确定性，不能消除热状态、allocator、CPU/主机资源影响。development CPU 官方评分与 main GPU 生成重叠，亦应披露可能的主机竞争。
- 旧图像复测不属于独立新测试。QA 只评最终答案，未审计长解释；COCO 自动指标不构成属性/关系/动作/遗漏的人工 Ground Truth。
- 保质结论必须要求预设所有适用门通过；数值不显著或某个自动分数变好都不支持“保证生成质量”。

非阻断完整性建议：final review 重新核对 SCORE_PROVENANCE.json 的 SHA（现有 resume loader 主要验证 aggregate/per-question）；开发已产生封印目前均匹配。不要修改已冻结评分器来改报告措辞；可以在独立补充审计/最终解释中加以上限制。

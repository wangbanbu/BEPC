# 独立实现审查（GPU 启动前）

结论：**PASS，无已发现的启动 blocker**。本审查未运行 GPU、未修改新旧方法源码；仅运行 CPU 数值/生命周期测试并核对冻结文件。

- 8 项单元测试全部通过：4,000 组随机耗尽预算扫描与原 `budgeted_scan` 结果及 budget-binding 计数一致；20,000 组非对称界覆盖独立公式；500 组实际 FP32 readout；EOS/padding/first-longest 边界；observer 退休与还原。
- 对称 `gamma_rescue=gamma_protection=.5` 的完整打分直接返回原 result，不重复排序 FP32 运算。正常阶段 `allowed` 复用旧路径。工程优化唯一行为是预算耗尽后的等价扫描及被动观察器退休。
- 耗尽扫描保留旧规则的行为，包括 EOS/可见边界后精确 token 匹配、padding 终止、首条最长前缀优先；不放宽 EOS mismatch。
- `update` 先完成原生 KV/输入更新与 observer commit，再累计 budget 并移除被动 hook；之后基类 tree/update 均检查 `self.epcc`，不会访问已退休对象。最终 stats 保留退休前计数，restore 可重复安全执行。
- 非对称全局修正范围为 [-.5,.625]；视觉预筛按 m_U 正负分别用 protection/rescue gamma。新完整评分只将 active rescue delta 乘 1.25，其他候选 score 不变。此实现有意保留旧 .5 FP32 计算路径，不声称与从头直接计算 .625 在每个 ULP 都相同。
- `run_experiment.py` 在每题与热身后比较 original/fast 的 raw/visible tokens、接受长度及路径；同题所有臂使用同一 input digest。真实 Target/Draft forward、tree mask、KV、EOS/最大长度继承旧模块。所有模型额外调用须由实际调用计数再次验证。
- 当前冻结配置所有源码 SHA 匹配；既有核心模块均为真实文件副本，旧路径没有经 symlink 被写入。旧文件及复制项完整性见同名 JSON。

尚需实验验证的项目：真实 GPU 上 original/fast 全样本 token/路径一致；旧运行与新 original/Strict 的输入/输出一致；内存和实际调用计数；单次计时不等于重复运行稳定性；自动质量非劣效需要按预设界判断，不能以不显著代替。

非阻断建议：`prepare_experiment.py` 的 `MAIN_READY` 先于 development 检查产生，因此实际 orchestration 必须在 development 成功及 parity 完成后才启动 main；配置与 runtime assertion 已提供逐题防线，最终报告应同时保存 development complete 证据。恢复运行若跳过已有 warmup，视为同一进程环境下延续的计时限制，不应当作新的等价/热身验证。下游指标评测仍需独立核对。

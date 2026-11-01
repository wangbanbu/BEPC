# MAT、长提示词与 AR 对齐审计入口

本轮继承[原长输出运行的 MAT、提示词与 AR 对齐审计](../../audit/MAT_AND_LONG_PROMPT_AUDIT.md)，其中的 token 计数、原生与有效 MAT 定义、512-token 预算和 AR 实现说明继续适用。

本轮重复次数为 **1**；原审计中涉及三次重复的调度与稳定性描述不作为本轮证据。单次运行范围、已保存数据复用及限制以[当前冻结实验协议](../EXPERIMENT_PROTOCOL.md)和[运行来源审计](RUNTIME_PROVENANCE.json)为准。

当前测试的 AR 与 Strict 输出一致性由最终完整报告逐条核对，不能从继承的开发检查推定正式测试全部一致。

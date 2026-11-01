# 官方 SPICE 资源恢复说明

首次正式评分中，LVSpec paper 的官方 SPICE/Stanford 解析器在 16GiB Java 堆上限下出现 OutOfMemoryError，跳过句子并最终退出失败；该次尝试没有产生有效的 SPICE 分数，也没有进入最终比较。原失败日志、输入和缓存已完整归档。

恢复仅将这一个官方评分进程的 Java 堆上限提高到 64GiB，保留相同 JAR、模型、线程参数、100条完整输入及参考答案，并使用全新缓存。没有重新生成 Caption，没有改变模型、阈值、样本、指标规则或任何解码计时。其他已成功评分直接复用。

恢复输出须满足100个图像ID齐全且无重复、无 OOM/忽略解析/批评分失败；输入字节、4200条推理记录、所有冻结源码配置和既有成功评分的 SHA 均独立核对。该资源调整属于处理执行故障，不是依据测试质量更改评价标准。LMDB 长字符串缓存写入警告与忽略解析分别审计，不能将两者混为一谈。

冻结评分包装器在读取相同输入的已验证结果时，仍在 provenance 中记录其计划的16GiB命令；此次结果实际由64GiB进程生成，以独立恢复审计中的真实命令和输出SHA为准。

证据：[资源恢复记录](../audit/SPICE_RESOURCE_RECOVERY.json)、[独立恢复验证](../audit/SPICE_RESOURCE_RECOVERY_VALIDATION.md)、[原失败流水线状态](../audit/SPICE_RESOURCE_PIPELINE_FAILED_STATUS.json)。

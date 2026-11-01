# 五任务长输出：单次实际解码比较

按用户新指示，本轮开发与正式比较只使用每个题目/方法一次生成（repeat0）。原三次开发调度在正式测试开始前停止，所有原始数据和已完成repeat1/2均保留作成本与来源记录；不进入本轮主比较。原开发repeat0全部复用，不重生成，仅补其缺项。身份与选择依据仅为固定题目/方法/repeat0，不看质量。

样本仍为MME/GQA/TextVQA/ScienceQA/COCO各20组开发、100组测试，MME每组完整双问题。因此9个开发配置×120题=1080条、7个主比较臂×600题=4200条。没有重抽图、换题或换答案。正式测试每题每方法恰好一次。每个需要新模型计算的task/shard每方法仅一次预热；已导入完整的开发task不加载模型和额外预热，实际旧/新预热单列。

Target/Draft、T=0、树、processor、seed、512-token上限、2048原生循环长度、停止、KV/mRoPE、attention与全部BEPC/CCVE参数保持原长输出配置。QA解释100–160词并以唯一终止Final answer行结尾；COCO100–160词详细描述。完整提示与问题输入直接只读引用原长输出JSON。没有强制最小长度或长度筛选。

核心Runtime从原封印长输出源码逐字节继承；新入口仅改变单次调度、已有repeat0复用及来源会计。每条复用记录保存原文件SHA/行号/原配置SHA及正规化字段，旧文件保持不变。所有问题固定GPU=group_ordinal%2，方法顺序仍原repeat0随机顺序。核心120条parity/48对照检查从同源码同输入的已通过证据继承，明确本轮没有重做parity，也不将其声称为主测重复稳定性。

七个主臂AR、原生ViSpec、共享工程Strict、BEPC算法变体、Margin4+CCVE、LVSpec论文比例、LVSpec开发适配保持。LV开发网格0/.3/.5/.7，paper先通过则保留，否则从更低且质量达标配置选最快；全失败显式Strict fallback。QA官方任务分数不降，MME任务总分及问题Accuracy不降，COCO SPICE≥95%Strict且CHAIRs不增。CIDEr长描述长度floor已知，原样计算但不作开发门。任何标准均不按测试结果调整。

主速度为可见文本tokens（不含EOS）总数/完整CUDA同步模型请求时间；另报decode-only、含EOS吞吐、请求时延、长度、峰值allocated/reserved及增量。原生Draft MAT=sum(a_r)/R，不含root；有效Draft MAT按首EOS/512预算裁切并按有效轮数汇总，completion包含第一个EOS。辅助1+有效MAT包含root，AR全部MAT为N/A。保留raw尾部/超预算与调用计数。AR显存仍含idleDraft。

QA只提取唯一终止Final answer，调用原封印官方评测；提取失败计0并保留分母、原文和原因。答案Accuracy不评价解释事实安全。COCO完整描述原样CIDEr-D/SPICE/CHAIR，不择短句、不用自动Judge当人工真值，SPICE/未定义与解析警告保留。

质量与时间都只用repeat0。统计仍5000次按image/group配对bootstrap；MME按category分层整组双问题采样。置信区间仅反映图像抽样，不包含重复计时波动，不能声称经过3次主测稳定性验证。BEPC六比较与LV适配两比较分别按task/metric Holm，保留所有结果。

原开发与测试图像物理独立；历史曝光限制原样保留（MME全部有旧运行，TextVQA测试3组、COCO测试2组），全项目未见图像泛化仍No-Go。原AR/native存在未定位输出差异，本轮独立报告而不预先归因浮点。未训练、融合、换验证规则或启动额外重复。

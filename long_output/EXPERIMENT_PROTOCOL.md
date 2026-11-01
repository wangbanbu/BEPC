# 长输出：五任务实际解码比较

本轮沿用干净ViSpec实验每任务20组开发、100组测试的全部图像、问题、顺序与官方答案，所有方法开发/测试均真实重复3次。MME保留完整双问题组；不因长输出失败、截断或质量结果删样。原短实验全部输入、配置、源码和结果保持不变。

## 处理与方法

唯一主要工作负载变化是短答案/单句提示改为长解释/详细描述，max_new_tokens由128改为512。QA保留原问题、ScienceQA hint及选项，要求约100–160词的有依据解释，并以唯一独立的Final answer行结束；COCO要求约100–160词详细描述。词数只是任务指令，不强制最小生成，不作有效性筛选。提示原文由harness/prepare_experiment.py固定。

Target/Draft checkpoint、T=0、seed、processor、原ViSpec树30/depth3/topk8/num_q2、max_length2048、after-block EOS和new_token>budget原生停止保持。官方Target实际KV容量4096，生成前检查prompt+512+树余量安全；不改位置、mRoPE或树注意力。

七个主臂：AR、原生ViSpec、共享工程优化Strict、BEPC算法变体、Margin4+CCVE、LVSpec论文比例、LVSpec开发适配。BEPC/CCVE的全部阈值、子空间和2/4 mismatch预算均原样迁移，LVSpec无2/4上限。15个runtime helper与短实验逐字节相同，runner只适配路径/元数据/会计和容量检查；原始upstream保持clean。

LVSpec在当前长输出20组开发上重新使用预设lambda={0,.3,.5,.7}，N10/PST/局部分支定义不变。论文.7通过质量门则保留，否则选较低且满足门的最快配置，若全失败显式关闭宽松。QA官方分数不降；MME acc+acc_plus及问题Accuracy均不降。COCO长描述已知受CIDEr-D长度Gaussian惩罚影响，因此事先固定开发门为SPICE>=.95Strict且CHAIRs不增；完整CIDEr照常输出但不作门。测试集绝不选参。

## 速度、MAT与显存

主速度为首EOS前/512预算内可见文本token总数（不含EOS）除以完整CUDA同步模型请求总时间。请求包含controller、视觉编码、prefill、所有Draft/Target及特征计算；PIL/processor/H2D单列。decode-only token/s、含EOS completion token/s、请求时延及长度另报。不同输出长度下token/s提高不代表请求更快。

官方Native Draft MAT=sum(a_r)/R，其中a_r是每轮严格/宽松最终路径接受的Draft token数，不含Target root。另报首EOS/预算裁切后的Effective Draft MAT=sum(effective_a_r)/R_effective，其completion含第一个EOS，不含其后尾部。辅助推进长度=1+Effective MAT，明确含root。全部MAT按轮数加权合并，不取每请求MAT简单平均；AR的MAT为N/A。保留raw EOS后尾部和预算外token计数。

记录实际Target/Draft/Vision调用、无额外Target forward、峰值allocated/reserved和增量。AR进程仍常驻未使用Draft，不能当作Target-only部署显存。请求间不empty_cache、不强制GC。

## 生成质量

QA仅按唯一、位于最后非空行的Final answer提取，随后调用上一轮封印的官方MME、GQA、TextVQA、ScienceQA评测。格式缺失、重复、非法、截断等失败计0并保留分母，分别记录解析失败与uncertain。QA解释的事实正确性不由答案Accuracy证明，不加入自动Judge或伪造人工标签。

COCO对完整长描述原样计算官方CIDEr-D、SPICE和作者CHAIR。CIDEr长度惩罚可能使长段落接近0，非零极小值用科学计数，主要质量图使用SPICE；各指标独立展示，不据其简单投票定义Harmful。保留SPICE解析警告/未定义结果，不静默换成短句或只选第一句。质量固定repeat0，三次确定性先核验，不挑较好重复。

## 正确性与统计

先做每任务前2开发组的真实GPU parity，检查原生/共享Strict、BEPC fast/reference、AR hidden-capture、LV disabled/Strict的输出与路径。主运行全部重复再次核验；AR/native先前存在输出差异，独立统计本轮差异，不未经诊断归因于BF16。源代码/KV/树mask合同与小规模parity不被称为完整数值因果定位。

开发3240请求，主测试12600请求，parity120请求，预热400次；两张L20按group_ordinal分片，每组固定GPU，方法顺序随机。每问题先平均3次时间，5000次image/group paired bootstrap；MME按category分层且整对抽样。质量、吞吐、两MAT、时间、显存等保留点式95%CI；BEPC对其余6方法和LV适配对论文/共享Strict两个比较家族按task/metric分别Holm。无显著差异不等于非劣效或等效。

开发/测试物理图像独立。历史曝光限制沿用原身份审计：MME全复用，TextVQA测试3组、COCO测试2组有旧记录；原freshness字段以补充勘误覆盖。不重新抽图，不声称全项目未见泛化。没有训练、融合、其他新方法或测试集调参。

# 干净 ViSpec 上的五任务解码比较

用户固定规模：每个数据集20组独立开发＋100组独立测试，所有方法开发/测试均3次重复。任务为MME、GQA、TextVQA、ScienceQA图像题、COCO Caption。MME以category/image的完整双问题组为单位；因此每阶段问题数分别120/600，总组数100/500。全部是预先抽样的子集，不能报告为完整官方split成绩。

## 源码与方法身份

ViSpec从 https://github.com/KangJialiang/ViSpec 全新clone，固定commit49a33ff0ca31e281b9dc8803fd6f3041c6c0e135。upstream目录不修改，每阶段保存git clean、所有tracked SHA及实际import的vispec模块路径；harness外置。BEPC/CCVE数学从前轮冻结实现移植，移植来源/摘要及仅局部import适配见audit。Target Qwen2.5-VL-7B与ViSpec-Qwen2.5-VL-7B checkpoint复用并固定身份。

主方法7臂：Target AR；原生ViSpec（原完整leaf logits及原末轮Draft构树）；Strict共享优化对照（compact logits＋跳过确定停止后的未使用Draft树）；BEPC算法变体；Margin4+CCVE；LVSpec-ViSpec论文比例；LVSpec-ViSpec开发适配比例。

BEPC保持wR=1、tau1、Urank8/Prank4、图像池64、合法普通文本W32、gamma.5/kappa1/eta1e-4、双向修正、每轮2/每回复4真实mismatch预算。Margin4+CCVE保持g<=4且CCVE<=-0.165220245718956及同2/4预算。两者参数不按本轮测试调节。工程优化只影响计算顺序/缓存/诊断或省去未使用末轮Draft构树，不能改变已提交tokens。

LVSpec保留论文视觉相关性Top10平均、每合法root→leaf分支相对排名、PST位置偏移容忍。N10与draft window10固定，但ViSpec depth3实际分支长度更短，使用lambda乘真实有效验证位置数floor映射；不能把absolute k7直接clamp成全放行。论文比例lambda.7；开发网格{0,.3,.5,.7}。lambda0仍保留PST，不等于Strict。视觉状态使用真实image_token_id行及最终norm后的原生hidden states；排除任意兄弟分支混池。官方代码与论文的prefill筛选、k7/K10差异及必要适配另见审计。LVSpec默认没有BEPC的2/4预算，不人为增加Margin或熵门冒充原方法，完整预算差异必须报告。EOS mismatch不宽松放行，padding与128边界统一处理。

## 提示词与输入

T=0；树total_token30/depth3/top_k8/num_q2；max_new_tokens128，max_length2048；原生after-block EOS与new_token>budget停止规则保留。主质量视图取128预算内首EOS前文本，raw/首EOS辅助保存，失败/截断/无法解析保留。

质量测试使用固定短答案格式：MME yes/no；GQA与TextVQA单词或短语；ScienceQA仅选项字母，问题/官方hint/choices，不向生成传入gold、lecture、solution或gold caption；COCO单句描述。系统词为You are a helpful assistant。原始ViSpec速度脚本长解释/OCR段落提示不用于本轮答案Accuracy；这一提示词变化必须说明。处理器固定use_fast=False、min_pixels=256*28²、max_pixels=1280*28²，所有方法完全相同，其他checkpoint默认参数不变。

输入在生成前按固定hash抽样，20/100按image分离；MME跨类别底层图像重复需检查。历史ID审计发现MME官方1187组全部已有历史生成，无法取得从未使用的120组；MME因此从原官方池重新固定抽样，开发/测试底层图像严格分离，完整报告历史复用，不声称全新未见图像泛化。其他任务优先排除历史ID；任何不可实现的历史独立性必须逐任务说明。自然可用样本池、旧记录/旧输入排除、缺图及来源选择完整保存；不以模型质量、信号或预期结论筛选。图像原字节、RGB、question_id/image_id/group_id、prompt与来源可追溯。参考答案与生成输入分文件，评分时才join。

## LVSpec条件调参规则

先在开发集判断论文比例是否过于宽松。QA主任务分数要求不低于原生Strict：GQA Accuracy、TextVQA官方VQA accuracy、ScienceQA多选Accuracy；MME还要求acc+acc_plus合计和问题Accuracy均不降。COCO代理要求CIDEr-D、SPICE各>=.95×Strict，CHAIRs不增。若lambda.7通过，保留论文比例；若失败，仅在更低比例中选满足约束且完整请求均值耗时最小者，完全并列取较小lambda。若全部失败，显式disable_relaxation并标注“开发集回退Strict”，不能作为成功LVSpec宽松策略。测试集不改参数。20开发组的代理约束不证明总体非劣效。

## 正确性与计时

CPU测试覆盖新增LVSpec rank/PST、正式任务解析与归一化、复用的子空间/readout、实际runner KV/mRoPE与EOS边界。GPU parity对每任务前2开发组测试原生/共享Strict、BEPC fast/reference、AR hidden capture和LVSpec disabled/Strict；保留数值审计与失败证据。主测试每图/组固定物理GPU=group_ordinal%2，随机方法顺序，3次重复；MME两个问题同GPU。模型初始化与processor不计入模型请求延迟；记录完整CUDA同步controller构造＋视觉编码＋prefill＋所有Draft/Target解码/在线特征成本。decode-only另报，不能混作端到端加速。输入预处理耗时另记，不把复用预处理谎称每方法重新测得的全链路延迟。

报告完整请求延迟、有效输出token/s、decode token/s、有效Draft接受数/验证轮、输出长度、实际调用、峰值allocated/reserved与增量。AR显存含常驻idle Draft；无额外Target forward。对象/hooks立即释放，禁止每请求gc/empty_cache制造不同计时口径。原生ViSpec基线不享受公共工程优化，Strict共享对照用于分离这种收益。

## 质量与统计

调用作者/官方实现：MME acc+acc_plus并分别perception/cognition；GQA exact-answer Accuracy；TextVQA官方VQA-style十答案归一化计分；ScienceQA官方多选Accuracy配固定解析器，解析失败算错、不随机猜；COCO CIDEr-D/SPICE/作者CHAIR，并明确不能推出完整事实安全。开发DF20、测试DF100分别计算，不直接比较绝对CIDEr。无需Judge或训练。

质量在重复确定性检查后用repeat0评分；重复若不同必须保留并报告，不能只挑较好一次。先按问题平均3次时延，再按image/group配对bootstrap5000次给点式95%CI；MME分层category并按完整图像问题组重采样。主比较BEPC对其余6方法，各任务/各指标Holm校正；LVadapt对LVpaper与Strict两比较独立预设family。不同任务不合并原始Accuracy或MME量纲，宏平均速度只作描述。所有指标/配置/源码/结果SHA可复核。100组是Pilot，不宣布微小质量差异为最终优势；自动指标不能替代人工Caption事实审核。

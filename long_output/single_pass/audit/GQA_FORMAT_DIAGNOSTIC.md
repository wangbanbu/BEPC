# GQA 长提示答案格式诊断

该诊断在查看冻结测试结果后执行，仅用于解释大小写格式敏感性；它不替换主表官方Accuracy，不重算开发选择，不更改标签，不进行新的显著性检验或推理。原评分逐题JSON和SHA均列于[机器记录](GQA_FORMAT_DIAGNOSTIC.json)。

变换仅为 `str.casefold()`：不去标点、冠词、空白，不作同义词映射。仅分析已有固定解析器成功提取的答案；提取失败仍保留。`大小写不符`表示原官方score=0且casefold后字符串相同；`仍不等`表示成功提取但casefold后仍不同。四列正确/大小写不符/仍不等/提取失败构成每行总数。

| Split | 实际评分臂 | N | 原官方正确 | 仅大小写不符 | casefold仍不等 | 提取失败 |
|---|---|---:|---:|---:|---:|---:|
| development | ar | 20 | 0 | 6 | 14 | 0 |
| development | bepc_algorithm | 20 | 0 | 7 | 13 | 0 |
| development | lvspec_ratio0 | 20 | 0 | 8 | 12 | 0 |
| development | lvspec_ratio0.3 | 20 | 0 | 4 | 11 | 5 |
| development | lvspec_ratio0.5 | 20 | 0 | 0 | 6 | 14 |
| development | lvspec_ratio0.7 | 20 | 0 | 1 | 7 | 12 |
| development | margin4_ccve | 20 | 0 | 8 | 11 | 1 |
| development | strict_shared | 20 | 0 | 7 | 13 | 0 |
| development | vispec | 20 | 0 | 7 | 13 | 0 |
| main | ar | 100 | 1 | 53 | 46 | 0 |
| main | bepc_algorithm | 100 | 2 | 49 | 49 | 0 |
| main | lvspec_adapted | 100 | 1 | 5 | 24 | 70 |
| main | lvspec_paper | 100 | 1 | 5 | 24 | 70 |
| main | margin4_ccve | 100 | 2 | 53 | 45 | 0 |
| main | strict_shared | 100 | 2 | 54 | 44 | 0 |
| main | vispec | 100 | 2 | 54 | 44 | 0 |

开发7个报告臂对应ar/vispec/strict_shared/BEPC/CCVE及同一个ratio0.7；后两者分别是paper与最终GQA-adapted的别名，并非重复开发测量。原四个ratio网格也全部保留。

Strict测试100题中官方正确2题，另54题仅大小写不同，例如Dress/dress、Snowy/snowy、Bed/bed。原2%不能直接解释为视觉问答能力只有2%；纯casefold诊断也不能自动宣称为原官方新成绩。应在下一轮冻结前解决提示格式与官方字符串比较的契约，先作无标签格式单元测试；本轮保持原始表与选择不变。

开发Strict、四个ratio均为0分，故paper .7通过“不下降”门仅表示0≥0，未提供有意义的保质证据；不能据此宣称该阈值在GQA已验证。测试也不能用于事后改选阈值。

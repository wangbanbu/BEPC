# AR / 原生 ViSpec 输出分叉诊断

仅分析现有token数组；无新增forward、评分、调参或源码修改。本文不建立AR/native等价性，不推断差异成因。

所有位置从0计数；“前10”是索引0–9，包含位置0。首分叉中位数仅在可见输出不同的pair中计算；一条序列是另一条的完整前缀时，记录的是末端边界，另行区分实际内容token替换。

| Task | Pairs | Visible different | First diff at0 | Within first10 | Median first diff | Content substitution | Prefix-end only | Common-prefix median (all) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mme | 200 | 127 | 0 | 11 | 37 | 127 | 0 | 64.0 |
| gqa | 100 | 76 | 0 | 10 | 40.0 | 76 | 0 | 51.5 |
| textvqa | 100 | 69 | 0 | 4 | 41 | 69 | 0 | 61.5 |
| scienceqa | 100 | 64 | 0 | 9 | 51.5 | 64 | 0 | 71.0 |
| coco | 100 | 91 | 0 | 14 | 39 | 91 | 0 | 46.0 |

此前“stop差异387”精确定义为：`eos_index`不相同，或`cap_reached`不相同。它表示观测到的停止位置/是否触顶不同，不能解读为采用了不同停止规则。各臂共享512-token有效预算与同一EOS；原生树block可在EOS/预算后含原始尾部，统一可见输出已按首EOS/预算裁切。

| Task | Stop outcome differs | eos_index differs | cap_reached differs | AR longer (all) | AR shorter (all) | Equal length (all) | AR longer among stop differences | AR shorter among stop differences |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mme | 118 | 118 | 0 | 56 | 62 | 82 | 56 | 62 |
| gqa | 68 | 68 | 0 | 30 | 38 | 32 | 30 | 38 |
| textvqa | 59 | 59 | 0 | 37 | 22 | 41 | 37 | 22 |
| scienceqa | 57 | 57 | 0 | 25 | 32 | 43 | 25 | 32 |
| coco | 85 | 85 | 0 | 40 | 45 | 15 | 40 | 45 |

合计：600对，427对可见输出不同，387对停止观测字段不同。共同前缀的分位数、区间分布、逐题token/停止字段及生成输入SHA见相邻JSON。

该描述性诊断无法解释差异来自何种数值或实现机制；不替代严格的AR/树形前向一致性定位。

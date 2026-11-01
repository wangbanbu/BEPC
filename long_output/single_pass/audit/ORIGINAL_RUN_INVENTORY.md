# 原三次重复运行库存

UTC observation: `2026-11-01T02:41:57.768447+00:00`。只读原运行；未读取 gold 内容；不以质量筛选。

可复用开发集 repeat 0：**1003 / 1080**；原 main：**0**。

| Task | Required repeat 0 | Reusable | Missing |
|---|---:|---:|---:|
| mme | 360 | 360 | 0 |
| gqa | 180 | 180 | 0 |
| textvqa | 180 | 180 | 0 |
| scienceqa | 180 | 180 | 0 |
| coco | 180 | 103 | 77 |

重复 1/2、预热和所有原始文件完整保留；不计入新的单次主比较。

原预热 190 次，436.694 秒。解析异常或部分行 0 条。

源文件 SHA、行偏移、行 SHA、候选复用映射、各方法覆盖和各 repeat 时间见 JSON。停机 PID 独立观察见 stop_evidence；SIGTERM 的操作证据由 root 会话与旧 RUN_STATUS 保留。

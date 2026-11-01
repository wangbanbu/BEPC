# SPICE 长文本缓存警告控制流审计

审计实际执行的 `spice-1.0.jar` 字节码，而非仅依赖邻近 Java 源码。完整 `javap -c -p` 输出、命令与SHA见同名 JSON 和 `.javap.txt`。

`LmdbTupleDB.putTransaction` 对单条 `LMDBException` 的处理只打印 `Could not cache item ... Caption may be too long`，随后继续下一个写入项并提交。`SpiceParser.loadTuplesFromDB` 已在内存中解析和存入 `newTuples`，调用缓存后在字节码227–230无条件执行 `captionTuples.putAll(newTuples)`。因此，这类持久缓存写失败不会仅因警告而删掉当前评分的caption或tuple。

这与 Stanford parser 的 `OutOfMemoryError`、`Will ignore and continue` 或评分批次异常不同。后者可能遗漏句子，必须使结果有效性检查失败；不能只检查Java退出码。本轮原LVSpec-paper日志包含OOM、忽略句子和NullPointerException，且无输出，不能使用；已成功五方法和开发九方法没有这些致命指示，但有上述缓存写警告。

该审计不证明自动场景图的事实或语义正确性，也不消除SPICE对长段落和不完整参考的局限。未调用GPU、未修改任何原始结果、评分源码或封印。

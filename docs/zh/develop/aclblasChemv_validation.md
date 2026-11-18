# aclblasChemv

`aclblasChemv` 在 Ascend 950PR 上计算 complex64 Hermitian 矩阵与向量的乘积：

```text
y = alpha * A * x + beta * y
```

矩阵采用列主序，支持上三角、下三角及正负向量步长。alpha、beta 为 Host 标量，A、x、y 为 Device 缓冲区；计算使用句柄绑定的 stream。

## 验证结果

在 Ascend 950PR、CANN 9.1.0 环境中：

| 项目 | 结果 |
|---|---|
| 专项精度 | 1002 / 1002 通过，参考为 Netlib BLAS/CBLAS 3.10.0 |
| 性能 | 200 / 200 通过，每例采集 60 个 kernel 耗时样本 |
| 标准构建与测试 | `bash build.sh --ops=chemv --soc=ascend950 --run`，1202 / 1202 通过 |

性能同时统计前 10 个及全部 60 个样本的算术平均值，两者均达到 GPU/NPU 耗时比不低于 0.4 的要求。应用先执行 5 次预热；采集工具为每个样本执行 5 次 replay 预热，每例额外预热 300 次。

## 内存检查

2026 年 11 月 7 日在 Ascend 950PR、CANN 9.1.0 上验证了提交 `3b6e573` 对应的源码。使用独立测试构建，Chemv kernel 编译时启用 `-g --cce-enable-sanitizer`，随后运行 `mssanitizer --tool=memcheck --leak-check=yes`。该环境下插桩构建和检查均成功。

| 分组 | 结果 |
|---|---|
| 接口契约（contracts） | 1 / 1 通过 |
| RegBase 上下三角尾块、中间块和大矩阵，含 lda padding | 7 / 7 通过 |
| 正步长与 padding、负步长、三类 alpha/beta 路径、上下三角补充尾块 | 7 / 7 通过 |

汇总为 **15 / 15 组通过**，共 138 次测试执行。14 个成功路径组均有 Chemv kernel 开始及无错误结束记录，`checked_kernels = 14`。本轮未报告内存错误或泄漏。实机归档中的汇总位于 `saved_results/memcheck_instrumented_final/summary.json`，同目录保存分组日志和 GTest XML。 [查看脱敏分组结果和实机日志摘录](aclblasChemv_memcheck_evidence.md)。

后续提交 `41e3137` 调整了测试封装和 CSV 行尾，host、kernel 和切分源码与上述实测版本一致。上表记录的是 `3b6e573` 的实机结果。


## 使用入口

- [接口说明](../../../blas/symv/README.md#aclblaschemv)
- [构建与测试](../../../test/symv/chemv/arch35/README.md)
- [实现设计](aclblasChemv_design.md)

## 采样与比较方法

每例先执行1次正确性检查和5次应用预热，再采集60个有效调用。`msprof op` 对每个样本执行5次replay预热，每例合计300次额外预热。采样数据为单kernel的 `Task Duration(us)`，报告同时保留前10次与全部60次的算术平均值。

精度按实部和虚部分别统计：`atol = rtol = 2^-13`，有限元素匹配率至少0.99，最大绝对误差不超过0.01，并执行特殊值及MERE/MARE检查。

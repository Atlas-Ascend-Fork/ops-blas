# CHERK arch22 测试


## CSV 构成

CSV 加载时检查总数、PF 数、非空及唯一 case_name；CSYR2 主表与 numerics 表还检查跨文件重名。
缺文件、空表、截断或筛选数量变化都会失败。当前构成：1224（PF 200）；默认精度 1024。

## FP64 数值参考

arch22 CSV/GTest 数值检查固定使用 FP64 参考，不依赖上游 presmoke，也没有回退到 FP32 golden 的开关。
FP32 输入精确提升为 FP64 后计算乘法和累加，参考结果保持 double 至误差比较；被测结果仍按 FP32 输出精度评判。
CHERK 使用 `cblas_zherk`，CHER2K 使用 `cblas_zher2k`，CSYR2 使用显式 double 计算。
任务书的 COMPLEX64 输出门槛为 `atol=rtol=2^-13`、匹配比例至少 0.99，硬误差上限为 `max(1e-2, 32×ULP)`，ULP 锚定最大绝对误差点的 FP32 golden。
非活动半区、padding 和 quick-return 字节检查仍与原始 FP32 输入比较。
日志中的 `TRIANGLE_CHECK reference=fp64` 表示实际使用的参考链。

小矩阵极值路径使用独立指数和 53 位有效数累加，避免有限输入在中间步骤提前溢出为 Inf/NaN，最终仍输出 FP32。
触发条件来自输入值域；共享实现位于 `blas/common/helper/scaled_scalar_arch22.h`。
`TC_FL_166` 及新增的极值系数回归使用同一 FP64 参考与 FP32 门槛。

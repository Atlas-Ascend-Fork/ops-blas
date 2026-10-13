# aclblasCgetrfBatched 测试说明

## 常规用例文件

`arch35/cgetrf_batched_test.csv` 保存全量用例按原顺序排列的前 50 项，供流水线默认执行。
测试入口按源码文件名定位同名 CSV，因此无需修改读取代码。
`arch35/cgetrf_batched_test_full.csv` 保留原有全部 1200 项用例（1000 项功能/准确性及 200 项性能场景），不由默认入口加载。
需要全量验证时，可在独立测试目录中用全量文件替换同名默认 CSV，再运行测试。
独立的 `cgetrf_batched_test_boundary.csv` 保存 56 项地址偏移测试。

## 自实现 golden 的语义依据

普通输入使用 [cgetrf_batched_golden.h](cgetrf_batched_golden.h) 中的 CPU 参考，从原始输入的独立副本计算，不使用设备输出，也不直接调用 OpenBLAS 或 Netlib 库。输入和运算为单精度复数，实部、虚部各为 float32。

- 分解及存储依据 [LAPACK CGETRF](https://www.netlib.org/lapack/explore-html/db/d04/group__getrf_ga478b8b7a93c3ded86200b70b41d2f384.html)：列主序、逐列消元和换行，L 的严格下三角与 U 的上三角原地保存，L 的单位对角线不存储；info 记录首次零主元的 1-based 位置，非奇异时为 0。
- 主元依据 [BLAS ICAMAX](https://www.netlib.org/blas/icamax.f)：在当前列剩余行中按 |实部|+|虚部| 取最大值，并列取第一行；PivotArray 记录 1-based 换行位置。
- 批量接口及空 PivotArray 关闭选主元的约定参考 [cuBLAS getrfBatched](https://docs.nvidia.com/cuda/archive/12.8.0/cublas/index.html#cublas-t-getrfbatched)。不选主元时直接使用当前对角元素，不生成主元数组。空规模和非法参数按本算子接口及官方用例预期返回，不照搬 LAPACK 的参数编号。

以上是语义依据，不承诺本 CPU 参考与任一库逐位一致；不同运算顺序、复数除法及极端幅值处理可能产生不同数值。本参考也不是高精度数学真值。当前常规用例和地址偏移用例均使用 CPU 参考，不包含独立的 NaN/Inf 特殊值对照测试。

## 精度判定

L/U 为浮点结果，不要求 bit-exact；info 和启用时的主元为整数，逐元素精确一致（bit-exact），包括奇异矩阵。仅参考 info>0 的矩阵跳过 L/U 数值比较，不能根据设备 info 自行豁免。

每张非奇异矩阵的实部、虚部分别检查，以下条件全部满足才通过：

1. 至少 99% 元素满足 |x−g|≤2⁻¹⁶+2⁻¹⁰|g|，且每个元素都满足 |x−g|≤max(0.01,32×ULP(g))。
2. 对进入相对误差统计的元素，e=|x−g|/(|g|+2⁻¹⁴)；MERE<2⁻¹³，MARE<10×2⁻¹³。10.0 是最大误差上限倍率，不是允许超限的元素个数或比例。等于上限不通过。

相对误差按公共框架顺序处理：完全相等、同为 NaN 或同号 Inf 时跳过；其余非有限值不匹配判失败；再排除 |g|<2⁻¹³ 的近零值。平均值只对剩余元素计算，集合为空且无不匹配时为 0。近零值仍受混合误差约束。ULP 为 float32 间距，非零有限值取 2^(max(⌊log₂|g|⌋,−126)−23)，零取 2⁻¹⁴⁹。

[精度检查实现](cgetrf_batched_precision.h) 同时调用两套检查，公共比较程序对应工程基础版本 `7f93ab5c73916c13f2b46f80c03a90badd65f9cf` 的 `test/frame/verify.h`。两套规则互不完全包含，不能择一使用。

## 地址偏移

[补充测试](arch35/cgetrf_batched_test_boundary.cpp) 包含 56 项地址偏移用例，使用有限数值输入及 CPU 参考，检查不同起始地址下的计算结果。对未豁免矩阵分别检查实部和虚部的两套误差；所有矩阵仍检查 info、启用时的主元、首尾保护区、列填充区和矩阵指针表。设备执行异常会阻断后续用例，未执行或作废记录不算通过。

验收包中的历史自测结果对应当时的测试范围和测量版本，不代表当前 PR 的测试覆盖或本次修改后的实机验证结果。历史结果文件不随源码提交。

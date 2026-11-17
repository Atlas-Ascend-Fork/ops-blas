# Csyr2 arch22 测试

使用仓库 CSV/GTest 框架。`csyr2_test` 执行精度、非法参数、重复调用以及未选半区、padding、guard 的按位回归；`csyr2_test_numerics.csv` 补充数值回归。CPU golden 使用 FP64 计算，保留 double 结果至误差比较。

alpha 支持 Host/Device 指针；Device 值经 Handle stream 读回，遵循该 stream 的依赖顺序。矩阵列距超出 32 位 DMA 字节步长时使用 64 位寻址通用核，不收窄合法 lda 范围。

在仓库根目录构建：

```sh
bash build.sh --soc=ascend910_9382 --ops=csyr2 --device=14
BLAS_DEVICE_ID=14 ./build/test/syr2/csyr2/csyr2_test
```


## CSV 构成

CSV 加载时检查总数、PF 数、非空及唯一 case_name；CSYR2 主表与 numerics 表还检查跨文件重名。
缺文件、空表、截断或筛选数量变化都会失败。当前构成：1222（PF 200）+ numerics 18；默认精度 1040。

## FP64 数值参考

arch22 CSV/GTest 数值检查固定使用 FP64 参考，不依赖上游 presmoke，也没有回退到 FP32 golden 的开关。
FP32 输入精确提升为 FP64 后计算乘法和累加，参考结果保持 double 至误差比较；被测结果仍按 FP32 输出精度评判。
CHERK 使用 `cblas_zherk`，CHER2K 使用 `cblas_zher2k`，CSYR2 使用显式 double 计算。
任务书的 COMPLEX64 输出门槛为 `atol=rtol=2^-13`、匹配比例至少 0.99，硬误差上限为 `max(1e-2, 32×ULP)`，ULP 锚定最大绝对误差点的 FP32 golden。
非活动半区、padding 和 quick-return 字节检查仍与原始 FP32 输入比较。
日志中的 `TRIANGLE_CHECK reference=fp64` 表示实际使用的参考链。


三角验证辅助实现统一位于 `test/utils/`，共享内核辅助实现位于 `blas/common/helper/`。
这些公共文件的修改按仓库既有规则触发全仓 CI。

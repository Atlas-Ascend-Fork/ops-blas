# CAXPY（Ascend 950PR）测试说明

## 范围

本目录的 CSV 驱动 GTest 覆盖 `aclblasCaxpy` 的参数校验、零长度、连续与非连续向量、正负步长、尾块和大规模连续 complex64 数据。精度 golden 由测试框架的 CPU 参考实现生成并分别比较实部和虚部。

## 环境

- Ascend 950PR
- CANN 9.1.0
- 从 `ops-blas` 仓库根目录执行

## 构建与功能自测

```bash
bash build.sh --ops=caxpy --soc=ascend950 --run
```

成功时输出应包含：

```text
[  PASSED  ] 15 tests.
[PASS] caxpy_test
```

## 性能复测

性能 case 均为 `complex64`、`incx=1`、`incy=1`，shape 分别为 1048576、2097152、4194304。测量应至少预热 60 次，并使用 ACL event 在 100 次有效采样上取平均。性能驱动是验收辅助工具，不作为仓库正式 ST 交付文件；提交前请按仓库规范移除该驱动，避免混入正式测试目录。

2026-09-13 的 Ascend 950PR 实测平均时延为 9.391 us、14.329 us、26.048 us（上述三个 shape 顺序）。

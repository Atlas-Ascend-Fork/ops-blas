# aclblasCsscal

单精度实数标量缩放单精度复数向量，原地执行 `x[i*incx].real/imag *= alpha`，对标 cuBLAS cublasCsscal。

| 产品 | 支持情况 |
|---|---|
| Ascend 950PR | 支持（arch35 实现；实机回归完成：1203/1203 通过，三条性能用例 12.1 / 18.9 / 31.6 微秒，全部优于标杆） |
| Ascend 950DT、其他产品 | 未验证 |

```cpp
aclblasStatus_t aclblasCsscal(aclblasHandle_t handle, int n,
    const float* alpha, aclblasComplex* x, int incx);
```

handle 为 ops-blas 句柄，alpha 指向 Host float，x 指向 Device complex64。正步长时，x 的物理长度至少为 `1+(n-1)*incx`。只更新选中的复数，间隔元素保持不变。

句柄为空优先返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；n≤0 返回成功；incx<0 返回 `ACLBLAS_STATUS_NOT_SUPPORTED`，incx==0 返回 `ACLBLAS_STATUS_INVALID_VALUE`（与 scopy/sswap/rotex/scalex 口径一致）；其余情况 alpha/x 为空返回 `ACLBLAS_STATUS_INVALID_VALUE`。alpha=1 可提前返回；alpha=0 仍执行乘法并保留特殊浮点语义。查询不到核数时返回 `ACLBLAS_STATUS_EXECUTION_FAILED`。

连续路径使用 AIV、单块 UB 原地处理，完整块与尾块分开，寄存器乘法使用满掩码；跳步路径使用 SIMT，地址计算为 64 位。无额外 GM workspace，UB 上限 248 KiB/核。调用为异步，读取结果前同步句柄流。

编译、运行及源码接入步骤见交付包根目录 README；详细实现见设计文档。三个性能目标为 13.57 / 21.05 / 43.02 微秒，均按预热后超过 50 次有效采样的平均值验收；实机 60 次采样均值 12.05 / 18.89 / 31.62 微秒（2026-09-21，3 轮均值），三条用例全部优于标杆。

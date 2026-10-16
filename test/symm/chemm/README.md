# aclblasChemm Ascend 950PR 自测

测试通过公共 `aclblasChemm` 接口调用真实 Device A/B/C，Host alpha/beta，并使用句柄绑定的 ACL stream。
Golden 使用系统 CBLAS `cblas_chemm(CblasColMajor, ...)`。

## 构建及运行

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:${LD_LIBRARY_PATH}
bash build.sh --soc=ascend950 --ops=chemm

./build/test/symm/chemm/chemm_test --gtest_filter=-*TC_PF*
./build/test/symm/chemm/chemm_test --gtest_filter=*TC_PF*
```

需要 CANN 9.1.0、Ascend 950PR、GTest 和 CBLAS/LAPACK 开发依赖。默认测试设备为 0，可在测试源码中通过 `TEST_DEVICE_ID` 调整。

## 用例组成

- `arch35/chemm_test.csv`：官方 1200 条 CSV，含 1000 条精度/接口用例和 200 条性能/内存用例。
- `arch35/gpu_baseline.csv`：200 条 GPU baseline。
- 独立 GTest 四项：空句柄；多句柄/多 stream 与用户 workspace 生命周期；偏移指针和保护区；最大有限 float 精度判据。
- required 精度口径为 1000 条官方非性能 CSV；测试二进制还会执行 4 项独立 GTest，因此预期总数是 1004。
- required 性能口径为 200 条 `TC_PF_*` 用例，并输出 200 条 `PERFORMANCE` timing 记录。

## 判定口径

- 复数结果分别验证实部和虚部，输出 `PRECISION,case,component,matched_ratio,max_abs_error`。
- 有限分量要求 99% 满足 `abs(error) <= 2^-16 + 2^-10 * abs(golden)`，且最大绝对误差不超过 `max(0.01, 32*ULP(golden))`。
- NaN 须对应 NaN，Inf 须同号；有限输出不允许匹配非有限值。
- C 的 padding 区域保持不变。
- 性能用例先通过精度，再预热 5 次，并用 ACL event 测量 60 次公共接口调用的平均 stream 耗时，单位微秒。
- 严格参考口径为 `GPU_us / NPU_us > 0.4`，200 条性能用例均需通过。
- 任务书四个硬性能 case 为 `TC_PF_1001` 到 `TC_PF_1004`，上限分别为 564.14、3824.68、547.60、3683.60 微秒。

## 20261008 实测结果

- required 精度：`1000/1000`
- required 性能：`200/200`
- timing rows：`200`
- 严格 `GPU/NPU > 0.4`：`200/200`
- 最低 ratio：`0.403752`
- 四项硬性能：
  - `TC_PF_1001`: 475.489 us
  - `TC_PF_1002`: 2778.590 us
  - `TC_PF_1003`: 455.534 us
  - `TC_PF_1004`: 2980.080 us

`MEMORY` 行记录性能用例 Device 分配前和首次调用同步后的空闲/总内存，差值包含输入、工作区和运行时分配，是快照而非峰值；内存查询发生在计时外。

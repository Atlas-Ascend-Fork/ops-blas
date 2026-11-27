# aclblasCrot A2/A3 自验证

本目录对应 11 月社区任务的 Atlas A2/A3（arch22）实现。目标环境为 CANN 9.1.0，
A2 使用 `--soc=ascend910b`，A3 使用 `--soc=ascend910_93`。性能验收设备为 910B3。

## 实现与接入

- Host：`blas/rot/arch22/crot_host.cpp`，先检查 handle、负 n、零 n，再检查步长、跨度和指针。
- Kernel：`blas/rot/arch22/crot_kernel.cpp`。连续路径每 tile 2048 个复数，
  双缓冲共使用 160 KiB UB。以 16×8 FLOAT32 块转置形成交替的 16 个实部/16 个虚部分组，
  通过掩码计算复数乘加，再逆转置写回，避免逐元素 Gather 和偏移表。
  x/y 原值完整保存在 UB，分离乘法与加减，不使用 FMA；跨步路径支持正负非零步长。
  路径与切分由步长、地址对齐及元素数量决定。
- 标量：Host 值保存在按值传递的 tiling 中；Device 指针直接下发，在绑定 stream 上读取。
- 公共接口 `include/cann_ops_blas.h` 已有正确声明，无需重复增加。
  `test/rot/crot/CMakeLists.txt` 已无条件调用 `ops_blas_add_gtest_tests`，
  由架构发现逻辑自动选择本目录，算子源文件也由工程自动收集。
- 测试入口复用仓库 `GetCasesFromCsv` 读取 CSV，按列名映射 Crot 参数，并复用 GTest 夹具、数据生成和精度校验。
- 不支持 x/y 区域相互重叠的调用；所有已分配 Device 输入须存活至 stream 执行完成。

## 用例覆盖与判定标准

`crot_test.csv` 共 2536 条，其中精度/边界 2336 条、性能 200 条。

| 来源 | 条数 | 说明 |
| --- | ---: | --- |
| 任务配套用例 | 1200 | 原 case_name 和数值保留，含全部 200 条 TC_PF |
| TC_GAUSS | 977 | 对原成功、正 n、x/y 都为均匀随机的精度用例补充正态分布版本 |
| TC_DIST | 288 | 36 种正负步长组合 × 4 种标量位置 × 2 种分布 |
| TC_BOUND | 13 | INT32_MIN、正负跨度溢出、大 n、n=1 极大合法步长、n=0 |
| TC_OFFSET | 48 | 不同 x/y 地址偏移、tile 边界及尾部长度 |
| TC_SCALAR | 10 | 标量 Inf/NaN、负实数 s，连续和跨步路径 |

`TC_DIST` 对 x/y/c/s 全部按均匀与正态各 50% 生成；实部、虚部和两向量采用独立种子。
均匀值域为 [-5,5]；正态使用框架 `GAUSS_NORM_5_2`，μ∈[-5,5]、σ∈[0.1,2]。
保留原有固定标量、极值、Inf/NaN 用例，作为分布测试之外的专门覆盖。
`cs_ptr_mode` 的 host/device/mixed/mixed_c_device 分别为 HH/DD/HD/DH。

按任务书 §2.4/§2.5，`TC_ED_123`（负 n）、`TC_ED_124/125/126`（零步长）
的预期状态为 INVALID_VALUE。任何预期失败的用例仅分配一个有效元素及保护区，
不按 n/步长申请缓冲。合法负步长仍按完整物理跨度分配。

每条成功数值用例比较 x/y 实部和虚部，并逐字节验证首尾保护区和步长间隙。
quick return 和失败用例验证全部缓冲保持不变。另有两个固定 GTest：
`ValidationPrecedence`、`DeviceScalarsFollowStream`。

精度使用共享 CPU golden 的有效参数计算路径，按任务书公式含 conj(s)；
arch22 的返回码独立验证。有限值调用框架 FLOAT32 混合容差验证，非有限值单独
统计并按任务书规则处理，不将 NaN/Inf 混入匹配率分母。

## 编译和精度验证（在目标环境执行）

在 ops-blas 根目录：

```bash
bash build.sh --ops=crot --soc=ascend910b
set -o pipefail
BLAS_DEVICE_ID=0 ./build/test/rot/crot/crot_test \
  --gtest_filter='-*/TC_PF_*' \
  --gtest_output=xml:crot_accuracy.xml 2>&1 | tee crot_accuracy.log
```

预期执行 2336 条 CSV 精度/边界用例及 2 条固定测试。检查 GTest 总数和零失败，
留存 XML 与日志中的有限值误差、匹配率及非有限值通过计数。
使用 A3 时更换 soc 参数后重新构建。

## 性能验证

按任务书 §3.3，使用 `msprof op` 采集单算子 Task Duration：

```bash
python3 test/rot/crot/arch22/verify_performance.py \
  --binary ./build/test/rot/crot/crot_test --device 0 --output ./crot_profile
```

默认逐条运行全部 200 个性能用例，单用例采集可加 `--case TC_PF_1001`。
通过条件为 200 条用例全部通过，每条取得 10 个有效样本。
输出目录必须是新目录，保存每条原始 msprof 文件、进程日志及 `results.csv`。
脚本按 case_name 关联 GPU 基线并校验 n/incx/incy；基线数据来自任务配套文件，
仅增加 case_name 列，避免相同 shape、不同标量位置的基线被覆盖。

脚本使用 `msprof op --warm-up=5 --launch-count=10 --aic-metrics=BasicInfo --kernel-name='*crot_kernel*'`，
匹配编译后的 C++ kernel 符号，仅采集验收需要的基本信息。
`CROT_PROFILE=1` 模式只发射 10 次算子调用，预热由 msprof 完成。
只有进程成功、测试确认 10 次成功调用、实际采集到 10 条有效 Task Duration(us)
时才计算均值；采样不足、异常数据和执行错误均判定失败。
每条需满足 `avg_us <= gpu_ms * 1000 * 1.25`，任何超限使脚本非零退出。
字段与预热/采集参数参见
[msOpProf 官方指南](https://github.com/Ascend/msopprof/blob/master/docs/en/user_guide/msopprof_user_guide.md)、
[OpBasicInfo 官方字段说明](https://github.com/Ascend/msopprof/blob/master/docs/en/user_guide/msopprof_performance_data.md#opbasicinfo-basic-operator-information)。

## 验证结果

2026-11-11 在 Ascend910B3、CANN 9.1.0、驱动 25.5.0 上完成自验证：
200/200 条性能用例全部达标，每条 5 次预热、10 个有效 Task Duration 样本，共 2000 个样本。
最小耗时余量为 18.03%（相对各用例的 GPU 基线 × 1.25 上限）。

| n | 实测均值（us） | 上限（us） |
| --- | --- | --- |
| 1048576 | 18.174 | 23.723 |
| 2097152 | 32.936 | 56.917 |
| 4194304 | 61.874 | 109.333 |

全量性能通过后，使用同一编译产物重新执行未过滤的 GTest：2538/2538 项全部通过，
其中精度/边界 2338 项、性能用例输出校验 200 项；没有跳过或禁用用例。

复现时优先使用系统自带的 `msprof op`。仅在设备号映射问题导致无法正常采集时，
手动编译并部署修订采集库，将设备号转换接口调整为
`aclrtGetLogicDevIdByUserDevIdImplOrigin`，按用户设备号获取逻辑设备号。

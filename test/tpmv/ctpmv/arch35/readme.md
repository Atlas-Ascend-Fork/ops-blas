# aclblasCtpmv（arch35 / Ascend 950PR）测试说明

## 测试环境

- 硬件：Ascend 950PR（arch35）
- CANN：9.1.0
- 数据类型：complex64（实部/虚部各 FP32）
- 测试框架：ops-blas 仓 test 目录 C++ GTest 框架，CSV 驱动用例（`ctpmv_test.csv`），golden 由 cblas（Netlib 复数实现 ctpmv）生成（`../ctpmv_golden.h`）

## 用例构成

`ctpmv_test.csv` 共 1200 条：

| 类别 | 前缀 | 条数 | 覆盖 |
|------|------|------|------|
| 基础 | TC_L0 | 24 | 12 组枚举全组合（uplo×trans×diag）× 小尺寸 |
| 尺寸 | TC_SQ | 23 | 1→2048 扫描（奇数/边界/非对齐/2 的幂±1） |
| 步长 | TC_INC | 72 | incx ∈ ±1/±2/±3 × 12 组组合 |
| 填充 | TC_FL | 12 | 均匀/全零/交替/极端值/Inf/NaN |
| 覆盖 | TC_CV | 96 | 中尺寸 × 12 组组合 |
| 边界 | TC_ED | 11 | n=0 no-op、UNIT 对角不读、空指针、非法枚举、负维度、零步长 |
| 扩展 | TC_EX | 763 | 尺寸池×组合×步长×填充确定性采样 |
| 性能 | TC_PF | 200 | 4 条任务书典型 case + 对数规模扫描 + 枚举×尺寸网格（均为 incx=1 连续访存） |

## 复现步骤

### 1. 编译

```bash
cd /path/to/ops-blas
bash build.sh --soc=ascend950
```

`ctpmv_test` 根据 `SOC_ARCH_DIRS` 选择对应目录的测试源；本目录的 950 用例仅在 arch35 目标下编入，A2/A3 的 arch22 测试独立保留。

### 2. 精度测试

```bash
# 方式一：直接跑 GTest 二进制（CSV 已随仓提供于本目录）
./build/.../ctpmv_test

# 方式二：使用任务配套脚本（自动编译+执行+解析逐条 PASS/FAIL，自动排除 TC_PF_ 前缀）
python verify_accuracy.py --repo /path/to/ops-blas --soc ascend950 --csv ./ctpmv_test.csv
```

精度判定（实部/虚部分别按 FP32）：

- 任务书口径：atol=2^-16、rtol=2^-10、matched_ratio≥0.99、max_abs_error≤1e-2
- 测试工程口径（CSV 列配置，与 stpmv 工程惯例一致）：mere_threshold=2^-13、mare_multiplier=10.0
- 非有限值输出与 golden 同为非有限值即视为一致（NaN 载荷差异不判错）

### 3. 性能测试

```bash
# GTest 输出为含 host 准备的保守上界（warmup + >50 次平均）
python verify_performance.py --repo /path/to/ops-blas --soc ascend950

# 精确 kernel 耗时（可选，用于对标任务书 §3.3 标杆）
msprof --application=./ctpmv_test --output=./prof_data
# 查看 op_summary_*.csv 中各 ctpmv kernel 的 Duration(us) 列
```

性能判定：NPU 平均单次耗时 ≤ gpu_baseline.csv 中 gpu_ms / 0.4（即对标 GPU H100 基线的 2.5 倍门槛）。

### 4. 异步与 workspace 回归

`aclblasCtpmv` 是接收Device缓冲区的异步生产接口；`ctpmv_npu_wrapper.h` 中的 `aclblasCtpmv_npu` 是接收Host数据、返回前读回结果的精度测试适配层。后者仅等待传入handle的stream（包括合法的默认stream），不做全Device同步。若提交失败但此前已有任务入队，仍须先完成该stream；同步失败时不读回也不提前释放可能仍在使用的设备缓冲区，由测试上下文清理。

CPU侧适配层回归不依赖NPU、CANN或第三方Python包，只需C++17编译器：

```bash
python3 test/tpmv/ctpmv/test_wrapper_stream.py
# 可选：用ASan/UBSan检查排队任务的缓冲区生命周期
CXXFLAGS='-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer' python3 test/tpmv/ctpmv/test_wrapper_stream.py
```

该测试覆盖显式/默认stream隔离、获取stream失败、部分提交失败、同步失败、读回失败、上传失败和快返回；使用延迟执行的ACL桩，不能替代设备数值测试。

`ctpmv_test_async.cpp` 随 `ctpmv_test` 自动编入，不修改原1200条CSV和精度阈值：

```bash
./build/.../ctpmv_test --gtest_filter='*Async*'
```

- `AsyncBackToBackAllPaths`：同一stream先提交D2D输入拷贝，再连续调用两次Ctpmv，最后才同步；覆盖tiny、nsmall、ns2、general、vector路径以及正负步长、T/C、UNIT。
- `AsyncUserWorkspaceCapacityAndOwnership`：用户workspace少1字节时返回容量不足；精确容量时连续调用成功，销毁handle后用户缓冲区仍由测试自行释放。
- `AsyncHandleDestructionCompletesQueuedWork`：不显式同步，直接销毁handle；检查销毁前完成待执行任务、输出与CPU golden一致。缓冲区晚于handle销毁释放。

这些用例检查stream顺序、连续复用和生命周期；仅靠运行结果不能证明Host从未阻塞，异步提交约束还需配合Host侧调用追踪或ACL模拟队列验证。下方数值为异步改动前的历史950PR实测，不代表新版本已完成设备复测。

## 实测结果参考（950PR 实机）

- 精度：1200 条全量 1001 OK / 0 FAILED（TC_PF 性能条目不参与精度判定）
- 性能（msprof kernel 实测，标杆=GPU 基线×2.5）：

| case | n | uplo | trans | incx | kernel 实测 (us) | 标杆 (us) | 比值 |
|---|---|---|---|---|---|---|---|
| 1 | 512 | UPPER | N | 1 | 31.0 | 44.89 | 0.69× |
| 2 | 1024 | LOWER | N | 1 | 57.2 | 88.46 | 0.65× |
| 3 | 2048 | UPPER | T | 1 | 93.7 | 184.2 | 0.51× |
| 4 | 4096 | LOWER | C | 1 | 776.0 | 1842.81 | 0.42× |

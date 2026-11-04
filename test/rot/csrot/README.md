# aclblasCsrot Ascend 950PR 自测说明

## 测试内容

Ascend 950PR 测试目录为 `test/rot/csrot/arch35/`，包括：

- `csrot_test.csv`：1206 条验收用例（1200 任务配套：1000 条精度/功能用例 +
  200 条 `TC_PF` 性能用例；另在文件尾部追加 6 条 `TC_EXS` 极端步长补充，总数 1206）；
  `TC_PF` 性能用例均连续访存（incx=incy=1）
- `csrot_test.cpp`：CSV 驱动的复数精度验证（实部/虚部分开统计），以及
  空 handle、n≤0 快速返回、null c/s、正/负/零步长及其混合、`c==1&&s==0`
  单位旋转不短路、NaN 传播、Inf+单位旋转逐位相等、host/device/mixed
  标量指针位置等专项测试；`TC_PF` 用例在 `CsvDriven` 中 early-return 仅做下发
  smoke + 返回码校验（不做 golden），正确性由精度套件覆盖，kernel 级计时统一由
  `csrot_benchmark` 采集；
- `csrot_benchmark.cpp`：设备事件计时，固定预热 20 次、有效采样 100 次，
  计时区间只覆盖 kernel 下发，并对任务书三个典型 shape 执行硬门槛断言；
- `csrot_npu_wrapper.h`、`csrot_golden.h`、`csrot_param.h`：Device 调用
  （含 c/s 空指针负向透传）、CPU golden 和 CSV 参数解析。

`test/rot/csrot/arch22/csrot_test.cpp` 是另一套独立程序（非 GTest、非
CSV 驱动），勿与上文 arch35 的 `csrot_test.cpp` 混淆：固定 n=1024、
c=s=cos/sin(π/4)、x≡1.0f / y≡2.0f 常量填充，单次 H2D→kernel→D2H 后
以 1e-5 相对误差与行内 golden 比对；无 CSV 用例、无 TC_PF 概念、不做
任何计时，也不覆盖空 handle、n≤0、null c/s、步长、NaN/Inf、cs_ptr_mode
等专项。上文关于 CSV 用例、TC_PF 同路校验、x/y seed 偏移与 benchmark
计时的描述均只针对 arch35（Ascend 950PR）；arch22 的 x/y 为不同常量，
本就相互独立，不涉及 seed。

精度 golden 显式实现 Netlib `csrot.f` 公式（实部/虚部逐通道独立、每步单次
舍入；负步长尾端起算、零步长串行累加、单位旋转不短路等 Netlib 语义不变），
与 NPU kernel 在全用例集逐位一致。精度判定走测试工程 MERE/MARE 口径
（MERE<2^-13、MARE<10×2^-13，实部/虚部分开统计，Inf/NaN 等值元素跳过）。

## 环境与依赖

验证环境为 Ascend 950PR、CANN 9.1.0。构建测试还需要 CMake、GTest
（仓库构建系统自动获取/配置）、BLAS 和 LAPACK 开发包。Ubuntu 22.04 可
安装：

```bash
apt-get update
apt-get install -y cmake libblas-dev liblapack-dev
```

## 编译

在 ops-blas 仓库根目录执行：

```bash
source /usr/local/Ascend/cann-9.1.0/bin/setenv.bash
rm -rf build
bash build.sh --soc=ascend950 --ops=csrot
```

预期生成：

```text
build/test/rot/csrot/csrot_test
build/test/rot/csrot/csrot_benchmark
```

## 精度与功能测试

运行 1006 条非性能 CSV 用例和 4 条专项测试（含 CsvCompositionAssert）：

```bash
./build/test/rot/csrot/csrot_test \
  --gtest_filter='-*TC_PF*' \
  --gtest_brief=1
```

如使用任务包中提供的精度脚本：

```bash
python3 /path/to/test_cases/verify_accuracy.py \
  --repo "$PWD" --soc ascend950 --skip-build --timeout 3600
```

预期汇总分别为 `1010 tests ... PASSED`（1006 精度 + 4 专项），以及脚本输出
`PASS=1006 FAIL=0`。

## 性能测试

本节仅适用于 Ascend 950PR（arch35）：arch22 目录没有 TC_PF 用例，也没有
`csrot_benchmark` 二进制，`arch22/csrot_test.cpp` 不做计时。

运行全部 200 条性能用例（kernel 级计时）：

```bash
./build/test/rot/csrot/csrot_benchmark --gtest_brief=1
```

只运行一个典型用例：

```bash
CSROT_BENCHMARK_CASE=TC_PF_1001 \
  ./build/test/rot/csrot/csrot_benchmark --gtest_brief=1
```

典型用例与门槛如下：

| case | n | incx | incy | 最大平均耗时 |
| --- | ---: | ---: | ---: | ---: |
| TC_PF_1001 | 1048576 | 1 | 1 | 21.69 us |
| TC_PF_1002 | 2097152 | 1 | 1 | 43.19 us |
| TC_PF_1003 | 4194304 | 1 | 1 | 119.38 us |

计时使用同一 stream 上的 ACL 设备事件，设备缓冲区按用例一次性分配并
初始化，计时区间只覆盖 100 次 kernel 下发；Host 侧内存申请、初始化和
结果处理不在计时区间。每条输出包含 `warmup=20`、`repeats=100`、
`avg_us`，三个典型用例还包含 `threshold_us`。

任务包自带 `verify_performance.py` 从 GTest 的整数毫秒 wall time 解析耗时，
其中还包含 Host 数据生成、Device 内存申请/拷贝和 golden 比对，属端到端
保守上界，不适用于本任务的 kernel 级判定；`csrot_test` 的 `TC_PF` 用例
本身仅做下发 smoke（不做 golden，正确性由精度套件覆盖）。
正式性能验收请使用上述设备事件 benchmark；这是任务书"先 warmup、有效
采样 >50 次"的直接实现。

## 内存说明

算子自身 workspace 为 0 Byte（纯原地运算，Host 侧无 `aclrtMalloc`）。
最大典型用例 `n=4194304` 时，一个 COMPLEX64 向量为 32 MiB，测试同时
分配 x、y 共 64 MiB；该输入/输出内存不属于算子 workspace。

# aclblasCher2 自测

适配 Ascend 950PR、CANN 9.1.0。源码位于 `blas/syr2/arch35/cher2_*`，
公共接口位于 `include/cann_ops_blas.h`。测试直接调用 handle/stream API，
golden 仅使用官方 Netlib BLAS 3.10.0 的 `cblas_cher2(CblasColMajor, ...)`。

## 用例

`arch35/cher2_test.csv`、`gpu_baseline.csv` 保留任务提供的原始内容。
C++ 装载和 Python 汇总分别校验 CSV 恰为1200行：1000条非 `TC_PF_` 精度用例、
200条性能用例，且名称非空、无重复。Python 另拒绝字段缺失/多余的行，
并固定核对2233项展开后的测试名称；CSV 丢行或缩小执行集合会直接失败。
验证脚本的本地故障注入检查可执行
`python3 -m unittest discover -s test/syr2/cher2 -p test_validation.py`，不依赖设备。

| 集合 | 数量 | 运行方式 |
| --- | ---: | --- |
| 原始精度 | 1000 | CSV 中非 TC_PF 行 |
| 原始性能及其数值检查 | 200 | TC_PF_1001~TC_PF_1200 |
| 正态分布补充 | 1000 | 对原始精度逐例添加 `_NORMAL` 变体 |
| 随机复数 alpha | 32 | TC_DIST_0~31，均匀/正态各 16 |
| 空 handle | 1 | Cher2Test.NullHandle |

原验收 GTest 共 2233 项。随机 x/y/A 的原始组为均匀 [-5,5]，配对组为正态
（实部/虚部独立采样，均值 [-5,5]、标准差 [0.1,2]）。特殊填充模式及 alpha 边界值
保持原始定义。TC_DIST 补足 alpha 的均匀/正态 50%/50% 随机覆盖，种子固定。
不需要重新运行 gen_csv.py；该脚本仅保留用于重新生成原始 CSV。

精度仅比较选定三角，实部、虚部分别输出 `[ACCURACY]` 记录：

- atol=rtol=2^-13，有限值 matched ratio >=0.99，最大绝对误差 <= max(0.01,32 ULP)。
- 非有限值单独计数，严格采用任务书的 NaN/Inf 等效规则，有限/非有限不等效。
- 另检查 CSV 的 MERE 与 MARE 阈值（近零豁免与分母 2^-14 与仓内框架一致）。
- 对角虚部为零；未引用三角、padding 使用 NaN payload/负零哨兵逐字节检查；x/y 只读检查。
- 边界状态、no-op 与空指针按 CSV 预期检查。

## 编译

先按 ops-blas 的安装指导准备 CANN 9.1.0、CMake、GTest 1.14 和 gcc/g++/gfortran。
为避免误用系统其他 BLAS，请构建并显式指定 Netlib 3.10.0：

```bash
curl -L https://github.com/Reference-LAPACK/lapack/archive/refs/tags/v3.10.0.tar.gz -o lapack-3.10.0.tar.gz
# SHA256: 328c1bea493a32cac5257d84157dc686cc3ab0b004e2bea22044e0a59f6f8a19
tar xf lapack-3.10.0.tar.gz
cmake -S lapack-3.10.0 -B netlib-build -DCBLAS=ON -DBUILD_SHARED_LIBS=ON -DBUILD_TESTING=OFF -DLAPACKE=OFF
cmake --build netlib-build --target cblas -j8
export NETLIB_BUILD=$(realpath netlib-build)
```

在 ops-blas 仓根目录执行：

```bash
source /usr/local/Ascend/cann/set_env.sh
export LINUX_INCLUDE_PATH=$ASCEND_HOME_PATH/include
export EAGER_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64
export LD_LIBRARY_PATH=$NETLIB_BUILD/lib:$LD_LIBRARY_PATH
cmake -S . -B build -DSOC_VERSION=ascend950 -DBUILD_TEST=ON -DTEST_NAMES=cher2 \
  -DCMAKE_BUILD_TYPE=Release -DREFBLAS_INCLUDE_DIR=$NETLIB_BUILD/include \
  -DREFBLAS_LIB=$NETLIB_BUILD/lib/libcblas.so -DREFLAPACK_LIB=
cmake --build build --target cher2_test -j8
```

## 精度运行

```bash
export BLAS_DEVICE_ID=0
python3 test/syr2/cher2/verify_accuracy.py --binary ./build/test/syr2/cher2/cher2_test --output ./cher2_accuracy
```

脚本检查退出码、GTest XML、完整 case 名单和实部/虚部指标；零用例、漏例、跳过均失败。
日志与 XML 保存在输出目录。仅调试时可直接用 GTest 的 `--gtest_filter`。

## 内存检查

任务书§3.4不要求提交内存验收报告；上游已有的内存安全检查入口仍保留。
使用插桩构建，沿用上述 Netlib 环境，在仓根目录执行：

```bash
cmake -S . -B build-sanitize -DSOC_VERSION=ascend950 -DBUILD_TEST=ON -DTEST_NAMES=cher2 \
  -DCMAKE_BUILD_TYPE=Release -DREFBLAS_INCLUDE_DIR=$NETLIB_BUILD/include \
  -DREFBLAS_LIB=$NETLIB_BUILD/lib/libcblas.so -DREFLAPACK_LIB= \
  '-DCMAKE_ASC_FLAGS=-g --cce-enable-sanitizer'
cmake --build build-sanitize --target cher2_test -j8
python3 test/syr2/cher2/run_memcheck.py --output ./cher2_memcheck
```

脚本调用原装 `mssanitizer --tool=memcheck --leak-check=yes`，按 ACL fixture 分进程：
第一组固定27项（空句柄、23条负向 CSV、512/1024/4096代表形状），第二组固定4项
`Cher2RegressionTest.*` 回归，内部遍历4字节对齐地址、正负大步长、特殊值及 AIV 分派边界。
144项混合参数仍在上述完整148项普通回归中检查；memcheck聚焦评审要求的边界队列，
将有Kernel执行的不同GTest suite分进程，避免ACL重建后复用工具的旧Kernel句柄。
两组均核对退出码、GTest实际数量和完整 XML，拒绝跳过/失败；检查每种 Kernel 的
`Start memcheck sanitizer` 与 `No error detected.` 正数配对记录及 WARNING/ERROR。
第二组必须同时有 SIMT 和 AIV 的检查记录，缺少插桩、漏跑、单有零退出码均不能通过。
各组日志、XML及 `summary.json` 保存在输出目录，实际结果以新提交的运行记录为准。

## 性能运行

性能只运行 200 项 TC_PF 用例，GPU 基线来自任务提供的 H100 原始 `gpu_ms`，转换为 us。
必须先通过上述完整精度检查。使用 `verify_performance.py` 采集 **msprof op 的 Task Duration(us)**，
不使用 GTest wall time、Event 或 Host 下发耗时。默认使用工具的 5 次 warmup，以及每例 60 个有效 Kernel 样本，取算术平均。

```bash
python3 test/syr2/cher2/verify_performance.py --binary ./build/test/syr2/cher2/cher2_test --output ./cher2_performance
```

脚本保留 msprof op 原始目录与日志，检查成功返回的 60 个有效样本、全部 200 条参数与 GPU 基线逐行一致，
计算 `GPU_us/NPU_mean_us >= 0.4`，任一漏例或未达标均返回非零。
性能模式由 CHER2_PROFILE 控制，不运行 CPU golden。范围采集时仅在范围前恢复 A；工具重放包含多次原地更新，
A 不参与性能判据，正常精度运行仍逐例对比单次更新。

## 参考

- [Netlib Cher2](https://netlib.org/blas/cher2.f)
- [MindStudio msOpProf 用户指南](https://github.com/Ascend/msopprof/blob/master/docs/zh/user_guide/msopprof_user_guide.md)


当前容器的 msopprof 26.1 设备查询会将公开设备号0转为宿主机物理号7，驱动采集接口仍需要容器设备号。
在该环境运行性能脚本时增加 `--container-device-fix`。脚本生成一个仅用于 profiler 的临时共享库，
让 `DeviceContext::GetRunningDeviceId` 使用公开 `aclrtGetDevice`；不改动 Kernel、运行频率或采集到的时间。
其他环境默认不启用此选项。使用 MSTX 范围重放，20 个 Kernel 一组、每例3组，共60个有效样本，
工具每个范围先预热5次；正式性能脚本每批至多16例，并保留全部原生采集结果。

## 上游集成与验收版本对应关系

本次以本人已验收提交 `4997e5145e50ea60ab7f7988308b0d6b40cb23a0` 的完整
Host、SIMT/AIV Kernel 和原始测试为基础接入当前 master。AIV Kernel 和 tiling
与原验收版本的 token 序列一致；Host 仅为静态检查提取等价的零步长判断，内联后
token 序列一致。按本次代码评审，SIMT 增加 x/y 运行时对齐分派：两者均8字节对齐
时保持原 `float2` 加载，否则逐 `float` 读取实虚部。A 始终使用标量读写。
公开 x/y/A 指针只要求 `aclblasComplex` 的4字节自然对齐；复数运算、舍入顺序、
参数校验及 Host 路径选择保持不变。
设计文档已通过 [社区 PR #2434](https://gitcode.com/cann/cann-ops-competitions/pull/2434) 合入。

保留上游已有的144条混合分布参数用例及4项补充回归，放在
`arch35/cher2_regression.cpp` 的 `Cher2RegressionTest` 中，覆盖特殊标量、
大步长、错位地址、混合特殊值和分派边界。原1200条CSV与空句柄检查已有本任务
测试覆盖，不重复注册。直接运行 GTest 时共2381项：本任务2233项，加保留回归148项。
`verify_accuracy.py` 只运行并严格核对原验收2233项，原始日志仍可用
`--summarize-only` 核对。原2233项与保留回归分别核对，集成验证还需执行以下148项
回归及上述内存检查，不能用旧验收报告替代这些新增验证：

```bash
./build/test/syr2/cher2/cher2_test --gtest_filter='*Cher2RegressionTest*'
```

200条GPU基线统一保存在 `gpu_baseline.csv`，数值及顺序与原验收基线一致。
旧版本 `run_op_profile.py` 等采集入口由本实现的 `verify_performance.py` 替代，
原精度入口 `run_self_test.py` 保留并转发至 `verify_accuracy.py`。
原验收报告对应上述固定提交；集成后的CI和验证状态应以新提交的实际记录为准。

为满足 CodeCheck，测试拆分过长函数，哨兵位复制使用 `memcpy_s`；Python 脚本
整理格式、使用 logging、拆分汇总函数并保留显式校验。原2233项日志/XML和
200项性能的12000份原始采样重新解析，输出 CSV 和汇总数值与原验收记录一致；
生成器在原配置及扩展配置下的用例参数和顺序一致。原报告及验收提交不变。

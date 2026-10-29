# aclblasCgemmStridedBatched ST

本目录采用仓库 GTest + CSV 框架，包含 1804 项 CSV 精度与接口用例及 3 项独立大步长回归，共 1807 项。CSV 包含：任务原始 1000 项、已有 708 项补充回归及 96 项 batch/UB 回归。覆盖 N/T/C、列主序、复数标量、广播与共享输出、padding、偏移、参数校验、零尺寸和 NaN/Inf 极值。原有 1708 项测试数据与已验收版本的对应精度用例逐格一致。

## 构建与运行

依赖 CANN 9.1.0、GTest、Netlib BLAS 和 LAPACK 开发包。使用实际设备编号；默认是逻辑卡 0。

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export EAGER_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64
export LINUX_INCLUDE_PATH=$ASCEND_HOME_PATH/$(uname -m)-linux/include
bash build.sh --soc=ascend910b3 --ops=cgemm_strided_batched --device=0
export LD_LIBRARY_PATH=$PWD/build:$ASCEND_HOME_PATH/lib64:$LD_LIBRARY_PATH
build/test/cgemm_strided_batched/cgemm_strided_batched_test
```

使用自定义安装路径时，先设置 CMake 缓存中的 `REFBLAS_INCLUDE_DIR`、`REFBLAS_LIB`、`REFLAPACK_LIB` 和 `CGEMM_SB_NETLIB_LIBRARY`，再执行上述构建命令。`CGEMM_SB_NETLIB_LIBRARY` 必须为真实 Netlib 共享库；默认搜索系统架构目录下的 `blas/` 目录。

## 参考结果与判据

精度程序隔离加载 Netlib，并核实 CBLAS 和 Fortran CGEMM 的实际符号提供者，输出参考库路径及配置时 SHA256。逐 batch 计算 CPU golden；共享输出 batch 按顺序执行。实部与虚部分别比较：atol=rtol=2^-13、匹配率≥0.99、最大绝对误差≤0.01；NaN 与同号 Inf 匹配。额外回归检查输出 padding、batch 间隔及保护区。

测试数据使用固定随机种子，实部与虚部独立生成；均匀与正态样本各占一半。快速返回及负向测试核对返回码和输出不变性。Device 缓冲区采用作用域管理，断言提前退出时释放。

## 文件布局

- `cgemm_strided_batched_param.h`：CSV 参数解析、确定性输入生成。
- `cgemm_strided_batched_golden.h`：Netlib golden、精度及 padding 检查。
- `arch22/cgemm_strided_batched_npu_wrapper.h`：Device 内存申请、拷贝及资源释放。
- `arch22/cgemm_strided_batched_test.cpp`、同名 CSV：GTest 入口及完整精度用例。

性能采样工具及验收材料保存在任务个人仓库固定版本 `4d9a267`，未混入本目录的 ST 交付。性能验收使用单上下文连续 60 次调用，主读数取第 6～15 次、稳态读数取第 11～60 次，逐项对比全部 200 个 GPU 基线。

## Batch 调度回归

`TC_RV_*` 的 96 项用例覆盖一般 Cube 路径与直接计算路径的 N/T/C 组合、复数 alpha/beta、紧密及带间隙/偏移布局、输入广播、共享 C，以及 alpha=0 时 A/B 为空的缩放。

- 直接计算和缩放由一次 kernel 下发处理全部 batch；输入经 DMA 暂存 UB，用 Gather/Mul/Add 等向量指令计算，再以 DMA 写回。共享 C 使用单核按 batch 顺序更新。
- UB 按32行分块，补充33/65行跨块和尾块回归，覆盖 N/T/C、非对齐 padding/保护区及共享输出。
- 输出 batch 独立的 Cube 路径对任意 alpha/beta 使用组内批量计算，pack、Cube 和 epilogue 在 Device 端遍历 batch，复用每个阶段的一次下发。
- Cube 的 Host 循环只在 workspace 容量不足以容纳全部 batch，或 C batch 重叠时拆组；前者复用有限 workspace，后者保持 beta 读取前一批结果的顺序。库管理的分组预算仍为 128 MiB，用户 workspace 不被扩容。

```bash
build/test/cgemm_strided_batched/cgemm_strided_batched_test --gtest_filter='*TC_RV_*'
```


## 大 leading dimension 回归

`CgemmSbWideStrideTest.LeadingDimensionBeyondDmaRange` 的3项独立用例分别覆盖 A、B、C 的大 leading dimension。
每次只在 Device 申请略大于4 GiB的带间隙矩阵，Host仅准备两列逻辑数据；验证超过32位DMA字节步长时仍按正确的64位列地址计算。
golden由同一Netlib参考实现计算，只在Host使用紧密存储的逻辑矩阵。
该组需要至少5 GiB空闲HBM，在设备无其他实验占用时运行；不能用跳过结果代替通过。

```bash
build/test/cgemm_strided_batched/cgemm_strided_batched_test --gtest_filter='*LeadingDimensionBeyondDmaRange*'
```

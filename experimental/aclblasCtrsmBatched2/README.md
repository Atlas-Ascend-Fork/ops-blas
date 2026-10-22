# aclblasCtrsmBatched

批量复数三角矩阵求解算子（complex64），基于 Ascend C 实现，对标 cuBLAS `cublasCtrsmBatched`。

## 功能描述

求解批量复数三角线性方程组：

- `op(A) * X = alpha * B`（side='L'）
- `X * op(A) = alpha * B`（side='R'）

其中 A 为复数三角矩阵，alpha 为复数标量，支持：
- Left/Right 左右乘模式
- Upper/Lower 上下三角
- NoTrans/Trans/ConjTrans 转置模式
- Unit/NonUnit 单位三角矩阵

## 接口定义

```c
aclblasStatus_t aclblasCtrsmBatched(
    aclblasHandle_t handle,
    aclblasSideMode_t side,
    aclblasFillMode_t uplo,
    aclblasOperation_t transa,
    aclblasDiagType_t diag,
    int64_t m, int64_t n,
    const std::complex<float>* alpha,
    const std::complex<float>* const aArray[], int64_t lda,
    std::complex<float>* const bArray[], int64_t ldb,
    int64_t batchCount)
```

## 架构设计

采用 MIX_AIC_1_2 混合核架构（1 Cube核 + 2 Vector核）：

```
AIV (Vector核): Panel 内三角求解（前代/回代）+ 数据格式转换（AoS<->SoA 转置）
AIC (Cube核):   Trail 区域 GEMM 更新（Matmul 库调用）
同步机制:       CrossCoreSetFlag/WaitFlag 跨核事件同步
```

**核心算法流程：**
```
for each panel:
  [AIV] LoadPanelA -> SolveInner(nb*nb) -> WriteBack Xneg
  [AIV->AIC] CrossCoreSetFlag(TRSV)
  [AIC] DirectRankK(GEMM) -> SetFlag(GEMM)
  [AIV] WaitFlag(GEMM) -> WriteBackPanelRows -> LoadPanelA(next) -> 下一 panel
```

**双 AIV 分列优化：**

大矩阵场景（kDim>=128 且 nColsAligned>=128）自动启用双 AIV 分列模式：
- 两个 AIV 核协同处理同一矩阵，各处理一半列
- 消除 AIV 空转，提升 AIV 利用率
- Panel solve 和 WriteBack 按列并行

**多核拆分优化：**

小 batch 场景（batch <= AI Core 数/2）自动启用多核拆分：
- 每个 batch 的列方向拆分到多个 AI Core 并行处理
- minSplitNCols=64，充分利用空闲核心

**转置优化：**

AoS<->SoA 格式转换采用 3 块 buffer 轮转设计：
- 动态 tileCols（根据可用 UB 空间计算，大矩阵 240~280 列/tile）
- Gather 偏移表解交织（支持任意非对齐尺寸）
- Duplicate 预清零 + dstStride 控制行步长（绕过 rightPadding < 32B 限制）
- padOn=false 时跳过清零，减少 PipeBarrier 开销

## 目录结构

```
aclblasCtrsmBatched2/
├── CMakeLists.txt
├── README.md
├── run.sh                              # 单用例快捷脚本（编译+生成+运行+校验）
├── op_host/
│   ├── ctrsm_batched_host.cpp          # Host 侧：参数校验、Tiling、Kernel 启动
│   └── ctrsm_batched_kernel_do.h       # Kernel 启动包装声明
├── op_kernel/
│   ├── ctrsm_batched_kernel.cpp        # Kernel 入口（MIX 模式，按 side/uplo/transa 分派四路径）
│   ├── ctrsm_batched_kernel_aic.h      # AIC Cube核实现（GEMM trail 更新）
│   ├── ctrsm_batched_kernel_common.h   # 公共常量和工具函数
│   ├── ctrsm_batched_tiling_data.h     # Tiling 数据结构定义
│   │                                   # ---- AIV 向量核按职责拆分的协作类 ----
│   ├── ctrsm_batched_kernel_aiv.h      # 编排类 CtrsmMixAivImpl<FORWARD,RIGHT>（含四路径别名）
│   ├── ctrsm_batched_kernel_aiv_cfg.h  # 共享派生配置 + UB buffer 指针
│   ├── ctrsm_batched_kernel_aiv_convert.h  # 复数 AoS<->SoA 转换/分块转置工具（路径无关）
│   ├── ctrsm_batched_kernel_aiv_canon_a.h  # A 矩阵规范化（补零/转置/共轭）
│   ├── ctrsm_batched_kernel_aiv_canon_b.h  # B 矩阵规范化与回写（左/右乘，模板 RIGHT）
│   ├── ctrsm_batched_kernel_aiv_canon_b_deinterleave.h  # B 矩阵 AoS 解交织实现
│   └── ctrsm_batched_kernel_aiv_solver.h   # Panel 三角求解（前代/回代，模板 FORWARD）
├── test/
│   ├── CMakeLists.txt
│   ├── ctrsm_batched_test.cpp          # 测试主程序
│   └── data/
│       ├── gen_data.py                 # 测试数据生成（输入矩阵 + golden 参考结果）
│       └── verify_result.py            # 精度验证
└── docs/
    ├── GPU_TEST                        # cuBLAS GPU 性能基准数据（320 case）
    └── perf_report_320cases_new.md     # NPU 性能报告（对标 GPU）
```

> AIV 实现按计算路径模板化为四个类型：`CtrsmLowerLeft` / `CtrsmLowerRight` /
> `CtrsmUpperLeft` / `CtrsmUpperRight`（`CtrsmMixAivImpl<FORWARD,RIGHT>` 的别名），
> 编译期消除 forward/right 分支；各职责类通过共享 cfg 与 UB buffer 指针协作。

## 编译运行

在本样例根目录下执行如下步骤，编译并执行算子。

- 配置环境变量

  请根据当前环境上CANN开发套件包的安装方式，选择对应配置环境变量的命令。
  ```bash
  source ${ASCEND_HOME_PATH}/set_env.sh
  ```

- 构建
  ```bash
  mkdir -p build && cd build
  cmake ..
  make -j8
  cd ..
  ```

- 样例执行
  ```bash
  # Step 1: 生成测试数据（输入矩阵 + golden 参考结果）
  python3 test/data/gen_data.py 64 64 32 0 0 0 0 1.0 0.0

  # Step 2: 运行算子
  #   参数: deviceId m n batch side uplo transa diag [alpha_re] [alpha_im]
  ./build/test/ctrsm_batched_test 0 64 64 32 0 0 0 0 1.0 0.0

  # Step 3: 验证精度
  python3 test/data/verify_result.py 64 64 32
  ```

  执行结果如下，说明精度对比成功：
  ```
  [Success] Case accuracy verification passed.
  ```

  也可用 `run.sh` 一键完成上述三步（编译+生成+运行+校验）：
  ```bash
  # 参数: m n batch side uplo transa diag [alpha_re] [alpha_im] [--skip-build]
  bash run.sh 64 64 32 0 0 0 0 1.0 0.0

  # 复用已有构建，alpha 使用默认值
  bash run.sh 64 64 32 0 0 0 0 --skip-build
  ```

## 参数说明

gen_data.py / ctrsm_batched_test 参数顺序：
```
m n batch side uplo transa diag [alpha_re] [alpha_im]
```

| 参数 | 取值 | 说明 |
|------|------|------|
| side | 0/1 | 0=Left, 1=Right |
| uplo | 0/1 | 0=Upper, 1=Lower |
| transa | 0/1/2 | 0=NoTrans, 1=Trans, 2=ConjTrans |
| diag | 0/1 | 0=NonUnit, 1=Unit |
| alpha_re | float | alpha 实部（默认 1.0） |
| alpha_im | float | alpha 虚部（默认 0.0） |

## 关键参数

| 参数 | 值 | 说明 |
|------|------|------|
| MIX_NB_SMALL | 16 | Panel 分块大小（kDim<=1024） |
| MIX_NB_LARGE | 32 | Panel 分块大小（kDim>1024） |
| LIM_GROUP | 16 | Xneg 分组大小 |
| FLOAT_ALIGN | 8 | 向量对齐宽度（float32，8 元素 = 32B） |
| TOTAL_AICORES | 20 | AI Core 总数（Ascend910B4） |
| minSplitNCols | 64 | 多核拆分最小列数 |
| SOC_VERSION | Ascend910B4 | 目标硬件 |
| dualAivMode | auto | kDim>=128 且 nColsAligned>=128 时自动启用 |

## 性能数据

320 Case 全量测试（对标 cuBLAS cublasCtrsmBatched）：

- 平均 GPU/NPU: **0.800**
- NPU 更快（GPU/NPU >= 1.0）: 109/320 个 case（34%）
- GPU/NPU 最大: 1.776（大矩阵场景 NPU 最多快 78%）

详见 `docs/perf_report_320cases_new.md`。

## 精度标准

相对误差阈值：`MERE < 2^-13 * 10 = 1.22e-3`

精度验证覆盖：
- 16 种 side*uplo*transa*diag 模式组合
- 矩阵尺寸 16~8192
- batch 数 1~248
- 复数 alpha 缩放
- 非对齐尺寸（17, 20, 31, 33, 227 等）
- 320 Case 全量精度通过

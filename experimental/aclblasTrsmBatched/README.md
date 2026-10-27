# aclblasTrsmBatched

批量三角矩阵求解算子（float32），基于 Ascend C 实现，对标 cuBLAS `cublasStrsmBatched`。

## 功能描述

求解批量三角线性方程组：

- `op(A) * X = alpha * B`（side='L'）
- `X * op(A) = alpha * B`（side='R'）

其中 A 为三角矩阵，alpha 为标量，支持：
- Left/Right 左右乘模式
- Upper/Lower 上下三角
- NoTrans/Trans 转置模式
- Unit/NonUnit 单位三角矩阵

## 接口定义

```c
aclblasStatus_t aclblasStrsmBatchedEx(
    aclblasHandle_t handle,
    aclblasSideMode_t side,
    aclblasFillMode_t uplo,
    aclblasOperation_t transa,
    aclblasDiagType_t diag,
    int64_t m, int64_t n,
    const float* alpha,
    const float* const aArray[], int64_t lda,
    float* const bArray[], int64_t ldb,
    int64_t batchCount)
```

## 架构设计

采用 MIX_AIC_1_2 混合核架构（1 Cube核 + 2 Vector核）：

```
AIV (Vector核): Panel 内三角求解（前代/回代）+ 数据格式转换
AIC (Cube核):   Trail 区域 GEMM 更新（Matmul 库调用）
同步机制:       CrossCoreSetFlag/WaitFlag 跨核事件同步
```

**核心算法流程：**
```
for each panel:
  [AIV] LoadPanelA -> SolveInner(nb*nb) -> WriteBack Xneg
  [AIV->AIC] CrossCoreSetFlag(TRSV)
  [AIC] DirectRankK(GEMM) -> SetFlag(GEMM)
  [AIV] WaitFlag(GEMM) -> 下一 panel
```

## 目录结构

```
aclblasTrsmBatched/
├── CMakeLists.txt
├── README.md
├── run.sh                            # 单用例快捷脚本（编译+生成+运行+校验）
├── op_host/
│   ├── trsm_batched_host.cpp         # Host 侧：参数校验、Tiling、Kernel 启动
│   └── trsm_batched_kernel_do.h      # Kernel 启动包装声明
├── op_kernel/
│   ├── trsm_batched_kernel.cpp       # Kernel 入口（MIX 模式分发）
│   ├── trsm_batched_aic.h            # AIC Cube核实现（GEMM trail 更新）
│   ├── trsm_batched_kernel_common.h  # 公共常量和工具函数
│   ├── trsm_batched_tiling_data.h    # Tiling 数据结构定义
│   │                                 # ---- AIV 向量核按职责拆分的协作类 ----
│   ├── trsm_batched_aiv.h            # 编排类 TrsmMixAiv（装配并驱动子类）
│   ├── trsm_batched_aiv_cfg.h        # 共享派生配置 + UB buffer 指针
│   ├── trsm_batched_aiv_transpose.h  # 分块转置引擎
│   ├── trsm_batched_aiv_canon.h      # A/B 规范化与回写
│   └── trsm_batched_aiv_solver.h     # Panel 三角求解与 Xneg 写回
└── test/
    ├── CMakeLists.txt
    ├── trsm_batched_test.cpp         # 测试主程序（单例 + config 批量模式）
    └── data/
        ├── gen_data.py               # 测试数据生成 + golden 参考实现（SciPy）
        └── verify_result.py          # 精度验证
```

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
  python3 test/data/gen_data.py 64 64 32 0 0 0 0 1.0

  # Step 2: 运行算子
  #   参数: m n batch side uplo transa diag [alpha]
  ./build/test/trsm_batched_test 64 64 32 0 0 0 0 1.0

  # Step 3: 验证精度
  python3 test/data/verify_result.py 64 64 32 0 0 0 0
  ```

  执行结果如下，说明精度对比成功：
  ```
  All 32 batches PASSED
  ```

  也可用 `run.sh` 一键完成上述三步（编译+生成+运行+校验）：
  ```bash
  # 参数: m n batch side uplo transa diag [alpha] [--skip-build]
  bash run.sh 64 64 32 0 0 0 0 1.0

  # 复用已有构建，alpha 使用默认值
  bash run.sh 64 64 32 0 0 0 0 --skip-build
  ```

- 全量测试

  使用 `run.sh` 配合脚本循环执行 320 个 GPU_test 用例：
  ```bash
  # 精度验证（逐 case 调用 run.sh --skip-build）
  bash run.sh 64 64 32 0 0 0 0 1.0 --skip-build
  ```

## 参数说明

gen_data.py / trsm_batched_test 参数顺序：
```
m n batch side uplo transa diag [alpha]
```

| 参数 | 取值 | 说明 |
|------|------|------|
| side | 0/1 | 0=Left, 1=Right |
| uplo | 0/1 | 0=Upper, 1=Lower |
| transa | 0/1 | 0=NoTrans, 1=Trans |
| diag | 0/1 | 0=NonUnit, 1=Unit |
| alpha | float | alpha 标量（默认 1.0） |

## 关键参数

| 参数 | 值 | 说明 |
|------|------|------|
| nb | 16/32/64 | Panel 分块大小（动态：kDim<256用16，<4096用32，≥4096用64） |
| LIM_GROUP | 16 | Xneg 分组大小 |
| FLOAT_ALIGN | 8 | 向量对齐宽度（float32） |
| SOC_VERSION | Ascend910B4 | 目标硬件 |

## 精度标准

相对误差阈值：`MERE < 2^-13 * 10 = 1.22e-3`

精度验证覆盖：
- 8 种 side*uplo*transa*diag 模式组合
- 矩阵尺寸 16~2048
- batch 数 1~256
- 非对齐尺寸（17, 31, 63, 127, 255）

## 与 CtrsmBatched 的区别

- **数据类型**: float32 vs complex64
- **ConjTrans**: 不支持（float32 没有共轭转置）
- **性能**: 理论上略快（数据量减半，无复数运算）
- **算法**: 核心 panel-update 算法相同

# aclblasLtMatmul — Atlas A2/A3

本目录补齐已有16参数公开`aclblasLtMatmul`的arch22路径，复用
`include/cann_ops_blasLt.h`，编入`ops_blasLt`；普通`ops_blas`自动收集排除此目录。
arch35走独立构建分支。本文描述该架构的接口、算法和资源约定。

| 产品 | 是否支持 |
|---|---|
| Atlas A2 系列产品 | 支持 |
| Atlas A3 系列产品 | 支持 |

## 接口语义

支持A/B/C/D FLOAT32、`ACLBLAS_COMPUTE_32F`、FLOAT32 HOST标量、ROW、单batch、
DEFAULT epilogue。实数OP_C等价OP_T，九种公开转置规约为四种实际转置。四路ld独立按
行宽解释。Create(ld=0) 后 GetAttribute(LD) 返回原始值 0，执行 ROW 计划时按
max(1, cols) 解析；需要预先计算 stride 时显式指定非零 LD。C==D要求有效布局一致。
与 arch35 的支持域及状态码差异见[公共接口说明](../../blasLt/README.md)。

校验顺序保持handle、描述符、标量、pointerMode、m/n零维、布局/类型/枚举、实际访问
指针及workspace。alpha=0或k=0不读A/B；beta=0不读C；退化beta=1异地精确复制旧C、
原地可no-op；beta非零且C空返回错误。显式stream原样传递，接口不隐式同步。

## 算法与设备内存

| 方案 | 容量合同 | 主要路径 |
|---|---|---|
| H0，ID `0x22000000` | 0字节额外GM workspace | Cube产品直接写D；原地beta非零用MIX保存旧C再合并；尾列串行向量处理 |
| H1，ID `0x22000001` | `4*m*ceil16(n)`字节 | 逻辑全N产品写padded R，再按独立C/D pitch合并；符合条件时直接写D |
| H2，ID `0x22000002` | H1产品容量 + 两个物理输入的NZ容量 + 16字节 | 先将A/B精确重排为NZ，再用FP32 Cube生成产品；非平凡alpha/beta使用R和epilogue |
| H3，ID `0x22000003` | `4*R+16`字节，`R=4*m*ceil16(n)` | 四路split-K产品及完整归并 |
| H4，ID `0x22000004` | R + A的NZ容量 + 16字节 | A重排、NZ/ND Cube及所需epilogue |
| H5，ID `0x22000005` | R | 短K矩阵的96×160 tile |
| H9，ID `0x22000009` | R | 小M转置矩阵的32列分块 |
| H10，ID `0x2200000a` | A/B重排、NZ产品及512B对齐余量，按`BuildFusedShortStorage`核算 | 短K矩阵融合路径，非平凡标量保留有序回退 |
| H11/H12，ID `0x2200000b`/`0x2200000c` | R | TT/NN短K矩形矩阵的独立tile |

`ceil16(n)=16*ceil(n/16)`。H1要求n>=16且对齐pitch可用INT32表示，否则heuristic只提供
H0。默认preference为32MiB；4096²的H1容量64MiB，测试wrapper明确设128MiB预算。
容量使用checked arithmetic，heuristic和launch共享模型；修改capsule的
`max_workspace_bytes`不能绕过检查。非空workspace只要求公共合同的16B对齐，不得与
活动输入输出重叠。此算子launch不分配内部设备矩阵工作区。

H2的物理矩阵`rows×cols`占用`4*ceil16(rows)*ceil8(cols)`字节，转置时先按
输入物理布局确定rows/cols。16字节余量用于把两个NZ平面起点向上对齐到32B；
调用方仍只需16B对齐。H2支持`0<k<=4096`，沿用128×128最大base tile的capsule，
不改变H0/H1容量。heuristic只在已验证的窄矩阵族、预算足够且容量不超过A+B+D
物理跨度总量1.2倍时优先H2；预算不足依次返回H1/H0。显式AlgoInit的容量仍由执行时
实际布局重新核算。所有路径都计入打包kernel，数值回退和退化缩放保留原规则。

调用方allocated和实际used分开记录。普通staged H1的used等于产品容量；
alpha=1/beta=0使用高阶Matmul原生ND输出，按逻辑N和真实ldd直接写D，包括非对齐尾列，
used=0。ND_ALIGN才要求填充输出行，不能将其限制套到ND。退化缩放和按Netlib顺序执行的数值回退也不使用R，
但显式H1仍须满足容量合同。测试canary和对齐余量属于测试开销，另列。

数值敏感的小矩阵、长K、非有限标量或大系数归约可使用Netlib FP32顺序的向量回退；
路由条件以最终冻结源码为准。H0 MIX用一组AIC与两个AIV，通过flag8/9/10协调旧C快照、
产品和合并，空半块完整握手。通过平台接口取得核数和片上容量，FP32 Cube关闭HF32。

H1长K的TN输出使用128×128网格时，若改用64×192能增加单轮内的计算核数，且任务数
不超过可用AIC，则选择64×192。仍按实际pitch计算完整K和逻辑N尾列，片上资源、
workspace容量及数值回退规则沿用H1。

capsule保留16字节`PackedAlgo`和magic `0xACBD1234`，校验算法ID、tile、policy、buffers、
splitK和flags，公开声明及配置接口ABI保留，不新增平行Matmul/AlgoCheck接口。
上述优化算法的支持域、预算和优先级以冻结host中的`Supports*`、`Build*Storage`及
`aclblasLtMatmulAlgoGetHeuristic`实现为准；预算不足保留H1/H0。H0/H1/H2的原容量
合同保持不变。H10等已启用路径纳入完整API计时，历史未采用融合候选的证据独立保留。

## 文件与测试工程

`arch22/lt_matmul_host.cpp` 负责参数校验、算法选择、workspace 和任务下发；
`lt_matmul_tiling.h` 定义按值传递的内部参数，`lt_matmul_kernel.h` 声明内部启动接口。
相关 kernel 按计算路径归并：

| 文件 | 计算路径 |
|---|---|
| `lt_matmul_cube_kernel.cpp` | ND Cube、MIX 原地 epilogue、TN 64×192 变体 |
| `lt_matmul_pack_kernel.cpp` | NZ 双侧/单侧打包及 16/64/80 行变体 |
| `lt_matmul_rect_kernel.cpp` | 矩形短 K、160 列带及 NZ 输入 Cube 变体 |
| `lt_matmul_split_kernel.cpp`、`lt_matmul_h9_kernel.cpp` | split-K 与长 K 流水 |
| `lt_matmul_mixed_nz_kernel.cpp`、`lt_matmul_fused_short_kernel.cpp` | NZ MIX 与短 K 融合流水 |
| `lt_matmul_fused_scale_kernel.cpp`、`lt_matmul_vector.cpp`、`lt_matmul_ordered.cpp` | 融合缩放、向量 epilogue 与有序数值回退 |

实现编入 `ops_blasLt`，测试使用仓库统一构建入口和 `blasLtMatmul_test` 目标。
GTest 的各回归文件独立编译；原 1200 行任务 CSV、200 行 GPU 表和 36 条补充 CSV 位于
`test/lt_matmul/arch22/`，精度与性能判据见[测试口径](../../test/lt_matmul/README.md)。

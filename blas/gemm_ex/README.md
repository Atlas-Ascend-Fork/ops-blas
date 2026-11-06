# GemmEx、SgemmEx、CgemmEx

计算列主序矩阵 `C = alpha * op(A) * op(B) + beta * C`。`op(A)`为m×k，`op(B)`为k×n，C为m×n。操作N保持原矩阵，T转置，C共轭转置；实数的C与T等价，复数的C会对虚部变号。

## 产品与类型

| 接口 | Atlas A2/A3（arch22） | Ascend 950PR/950DT（arch35） |
|---|---|---|
| aclblasGemmEx | FLOAT32，三个矩阵类型均为ACL_FLOAT | 沿用arch35实现的类型组合 |
| aclblasSgemmEx | FLOAT32，三个矩阵类型均为ACL_FLOAT | FLOAT32 |
| aclblasCgemmEx | COMPLEX64，三个矩阵类型均为ACL_COMPLEX64 | 保留已有实数/复数及半精度组合；统一公共aclDataType接口 |

arch22支持Atlas A2/A3系列，实机自测范围与当前提交的最终结果见自验收报告；任务允许覆盖一种款型，未实测A3时不能把架构支持写成A3测试通过。arch22的GemmEx只接受`ACLBLAS_COMPUTE_32F`和`ACLBLAS_GEMM_DEFAULT`。其他合法dtype或不支持的组合返回NOT_SUPPORTED，非法dtype数值返回INVALID_VALUE，不隐式转换为FP32。

arch35 GemmEx保持既有组合：FP16/FP16/FP16配`ACLBLAS_COMPUTE_16F`或`ACLBLAS_COMPUTE_32F`，BF16/BF16/BF16、FLOAT/FLOAT/FLOAT配`ACLBLAS_COMPUTE_32F`，FP8 E4M3FN/E5M2输入组合配FP16输出及`ACLBLAS_COMPUTE_32F`。这些既有路径不属于本次arch22实机验证结果。

## 公共接口

声明统一位于`include/cann_ops_blas.h`。GemmEx为19参数；SgemmEx和CgemmEx为17参数，包含三矩阵dtype，不含computeType/algo。旧15参数SgemmEx调用方需要迁移。950旧CgemmEx私有dtype数值0/1/2/3分别迁移为公共ACL_FLOAT/ACL_COMPLEX64/ACL_FLOAT16/ACL_COMPLEX32；两种枚举的整数语义不同，调用方须更新类型并重新编译，不能原值强转。950内部存储枚举、支持组合和既有计算实现保留；本次A2实测不等于950硬件验证。

```cpp
aclblasStatus_t aclblasGemmEx(
    aclblasHandle_t handle, aclblasOperation_t transA, aclblasOperation_t transB,
    int m, int n, int k, const void* alpha,
    const void* A, aclDataType Atype, int lda,
    const void* B, aclDataType Btype, int ldb, const void* beta,
    void* C, aclDataType Ctype, int ldc,
    aclblasComputeType_t computeType, aclblasGemmAlgo_t algo);

aclblasStatus_t aclblasSgemmEx(
    aclblasHandle_t handle, aclblasOperation_t transA, aclblasOperation_t transB,
    int m, int n, int k, const float* alpha,
    const void* A, aclDataType Atype, int lda,
    const void* B, aclDataType Btype, int ldb, const float* beta,
    void* C, aclDataType Ctype, int ldc);

aclblasStatus_t aclblasCgemmEx(
    aclblasHandle_t handle, aclblasOperation_t transA, aclblasOperation_t transB,
    int m, int n, int k, const aclblasComplex* alpha,
    const void* A, aclDataType Atype, int lda,
    const void* B, aclDataType Btype, int ldb, const aclblasComplex* beta,
    void* C, aclDataType Ctype, int ldc);
```

arch22的alpha/beta为Host标量指针；A/B/C为Device矩阵指针。arch35 CgemmEx沿用上游的Host/Device标量解析。复数按`aclblasComplex`的real/imag交错存储，lda/ldb/ldc按复数元素计数。handle通过`aclblasSetStream`绑定stream，接口成功返回表示已按该stream提交，读回结果前需要同步。

## arch22参数与边界行为

| 参数 | 要求 |
|---|---|
| handle | 已创建的有效句柄；nullptr返回HANDLE_IS_NULLPTR |
| m、n、k | 均非负 |
| transA、transB | N、T或C；其他值返回INVALID_ENUM |
| lda | transA=N时至少max(1,m)，否则至少max(1,k) |
| ldb | transB=N时至少max(1,k)，否则至少max(1,n) |
| ldc | 至少max(1,m) |
| alpha、beta | 非空Host指针 |
| A、B | k>0且alpha非零时必须非空 |
| C | beta非零时必须非空 |

校验顺序为handle、维度/leading dimensions、alpha/beta、dtype、转置枚举、矩阵指针、GemmEx的computeType/algo，最后才判断no-op。leading dimension和其他参数非法返回INVALID_VALUE；合法但不支持的computeType/algo返回NOT_SUPPORTED。

- m=0或n=0在校验完成后成功返回，不计算。
- k=0或alpha=0只处理`C=beta*C`，不读取A/B。beta=1不改C，beta=0不读取原C，其他beta按逻辑元素缩放。
- beta=0且C=nullptr直接成功返回，全程不读写C，包括k=0分支。
- 只写C的逻辑m行，leading dimension中的padding保持原值。支持列主序leading dimension，不提供其他任意步长视图或广播。

## 实现与精度

三个公共接口分别调用各自的分派入口，专用kernel分文件保存。GemmEx与SgemmEx薄包装仍位于同一host文件，通用基础代码和部分fallback在gemm_ex内部复用，不改写普通aclblasSgemm；不能由一个接口通过推断另一个接口通过。根据公开尺寸、转置、leading dimension及标量值选择分块、打包和缩放路径，不按测试用例编号或GPU表选择实现。主机入口及分派集中在对应host文件，删除仅承载薄转发函数的独立源文件。主机侧共同规划通过显式kernel函数表绑定GemmEx/SgemmEx各自的专用入口；共享辅助代码按host规划、device tile、DMA写回三类放入私有头；SgemmEx专用符号直接声明，不使用宏重命名。设备kernel入口、参数布局和各入口的同步编译选项分别保留。同算子、同编译配置的计算与输出kernel按职责合并在同一cpp，仍是独立kernel函数、按原顺序调用。

arch22共19个文件（13个cpp、6个私有头）。主要设备实现分组如下：

| 文件 | 职责及编译边界 |
|---|---|
| `gemmex_short_nn_kernel.cpp` | GemmEx短NN计算与输出，显式同步编译 |
| `sgemm_parity_nn_kernel.cpp` | SgemmEx独立短NN计算与输出，显式同步编译 |
| `sgemm_short_nn_kernel.cpp` | 短NN回退计算与输出，保留默认同步编译 |
| `sgemm_backend_kernel.cpp` | SgemmEx原生后端及small-TT、stagger路径，保留相同宏与默认同步编译 |
| `cgemm_longk_kernel.cpp` | CgemmEx长K打包与矩阵乘，两个kernel分别启动 |
| 其余host、fallback及复数主体 | 保留完整职责，不将三个算子的专用后端合成一个kernel |

不再保留对应的六份输出/打包/小形状独立源文件；构建从干净源码目录收集，避免旧cpp残留造成重复定义。 已清理没有调用点的历史DMA copy、融合实验和旧短K/复数转换入口；活跃分派继续保留，不以源码入口数量替代性能或正确性验证。

对于|alpha|≤1的非单位标量，超过256³个乘加项的问题使用通用后端及alpha/beta epilogue；小输出direct限制k≤4096。|alpha|>1仍保留补偿计算：先计算FP32乘积再放大可能超出golden最大误差限制。该数值回退保证原精度语义，但大幅值alpha的大矩阵加速仍待验证，不能视为所有非单位alpha的性能问题均已解决。

短NN按两块368行tile组成偶/奇核对选择连续形状族（368<m≤736、n≥16、16<k≤32、紧凑NN且alpha=1/beta=0）；运行时还要求grid不超过可用Cube核数、输出列组不超过AIV核数。原721～736只是先前调优窗口，现已删除；不将该窗口解释为硬件边界。

GemmEx短NN使用独立的输入/计算与输出kernel，在适用的短K形状族中按行核奇偶交错两个K段的搬运和乘加次序，每个乘积仅计算一次；浮点累加次序改变后仍按同一Netlib判据检查。SgemmEx具有独立短NN、小N转置及输入就绪同步路径；适用短K形状族的独立kernel也采用同样的按行K段交错方案，两个接口仍分别验证。短K使用NZ输出和批量DMA恢复布局，中等K使用多K步NZ流水；全部必要搬运、计算和输出阶段都计入接口耗时。

复数采用直接向量、小输出四实乘积和实数块矩阵展开路径，长K适用形状使用独立打包和矩阵乘kernel。展开处理转置/共轭并显式清零padding；乘积完成后恢复实虚交错并应用alpha/beta。单核流水使用相应事件同步，阶段间按同一stream有序执行。workspace来自handle的既有机制。

复数短NN形状族有独立的Cube路径：公开K在17～32范围、M在512～1536范围且N在128～1024范围时，将展开后的至多64个实数K坐标一次装入L0，输出仍写回原padding布局，再由通用输出步骤恢复公共C。模式3的虚部符号处理独立采用奇数位置掩码的标量乘法；这项优化不改变短NN的分块。两处实现按布局和公开参数分派，互不替换，也不改变GemmEx/SgemmEx的专用后端。

arch22不以HF32/FP16替代FP32输入计算。测试golden为Netlib BLAS/CBLAS3.10.0：实数使用cblas_sgemm，复数使用cblas_cgemm，实/虚分别比较。任务精度为rtol=atol=2^-13、matched_ratio>=0.99，最大绝对误差不超过max(0.01,32×ULP(最大误差点golden))；非有限值和alpha=0的EXACT规则单独检查。

## 测试与复现

原始三组CSV各包含1000条精度/边界用例和200条性能用例，保留其数据、种子和GPU基线。构建、golden版本、msprof op采样、原始时序解析及报告说明见[arch22自测步骤](../../test/gemm_ex/arch22/README.md)。实现和测试结果只以最终报告锁定的源码、依赖、动态库及实际设备为准。

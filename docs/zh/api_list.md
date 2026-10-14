# 算子接口

## 使用说明

ops-blas 提供基于 C 的 BLAS 标准接口，以及轻量化 GEMM / 矩阵变换接口，便于在 NPU 上高效完成线性代数计算。

- **头文件/库文件**

  调用接口时，需引用依赖的头文件和库文件。头文件默认位于 `${INSTALL_DIR}/include`，库文件默认位于 `${INSTALL_DIR}/lib64`，具体文件如下：

  - 头文件（推荐引用总头文件）：
    - [cann_ops_blas.h](../../include/cann_ops_blas.h)：aclBLAS 计算接口与 aclBLAS Helper 接口
    - [cann_ops_blas_common.h](../../include/cann_ops_blas_common.h)：公共类型与枚举定义
    - [cann_ops_blasLt.h](../../include/cann_ops_blasLt.h)：aclBLASLt 计算接口与 aclBLASLt Helper 接口
  - 库文件：
    - `libops_blas.so`：aclBLAS 库
    - `libops_blasLt.so`：aclBLASLt 库

  `${INSTALL_DIR}` 表示 CANN 安装路径。

## aclBLAS Datatypes Reference

[cann_ops_blas_common.h](../../include/cann_ops_blas_common.h) 定义 aclBLAS / aclBLASLt 共用的类型与枚举。

### aclblasStatus_t

函数状态返回类型。所有 aclBLAS / aclBLASLt 库函数均通过该类型返回执行结果。`aclblasLtStatus` 为同类型别名。

| 取值 | 含义 |
|---|---|
| `ACLBLAS_STATUS_SUCCESS` (0) | 函数执行成功。 |
| `ACLBLAS_STATUS_NOT_INITIALIZED` (1) | 库未初始化，通常因未先调用 `aclblasCreate` / `aclblasLtCreate`，或 CANN 运行环境未就绪。 |
| `ACLBLAS_STATUS_ALLOC_FAILED` (2) | 库内部资源分配失败（如 device/host 内存申请失败）。 |
| `ACLBLAS_STATUS_INVALID_VALUE` (3) | 传入了非法或不支持的参数值（如维度为负、指针为空且不允许为空等）。 |
| `ACLBLAS_STATUS_MAPPING_ERROR` (4) | 访问设备内存空间失败。 |
| `ACLBLAS_STATUS_EXECUTION_FAILED` (5) | 算子在设备侧执行失败（如下发失败、stream 同步失败等）。 |
| `ACLBLAS_STATUS_INTERNAL_ERROR` (6) | 库内部操作失败。 |
| `ACLBLAS_STATUS_NOT_SUPPORTED` (7) | 当前平台、数据类型或参数组合下，该功能尚未实现或不支持。 |
| `ACLBLAS_STATUS_ARCH_MISMATCH` (8) | 设备架构与当前实现不匹配。 |
| `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` (9) | 传入的 handle 为空指针。 |
| `ACLBLAS_STATUS_INVALID_ENUM` (10) | 传入了不支持的枚举取值。 |
| `ACLBLAS_STATUS_UNKNOWN` (11) | 后端返回了库未识别的状态码。 |

### aclblasFillMode_t

指定对称矩阵、三角矩阵或 packed 矩阵使用上三角还是下三角部分。取值与 Fortran BLAS 中 `U` / `L` 参数语义一致。

| 取值 | 含义 |
|---|---|
| `ACLBLAS_UPPER` (121) | 使用矩阵的上三角部分。 |
| `ACLBLAS_LOWER` (122) | 使用矩阵的下三角部分。 |

### aclblasDiagType_t

指定三角矩阵的对角线元素是否视为单位阵（对角元为 1、不参与读写）。取值与 Fortran BLAS 中 `N` / `U` 参数语义一致。

| 取值 | 含义 |
|---|---|
| `ACLBLAS_NON_UNIT` (131) | 非单位对角线，对角元素从矩阵数据中读取。 |
| `ACLBLAS_UNIT` (132) | 单位对角线，对角元素视为 1，调用方无需显式存储。 |

### aclblasSideMode_t

指定对称矩阵乘法（symm）等运算中，对称矩阵位于乘积的左侧还是右侧。

| 取值 | 含义 |
|---|---|
| `ACLBLAS_SIDE_LEFT` (141) | 对称/三角矩阵在乘法左侧（op(A)·B 形式）。 |
| `ACLBLAS_SIDE_RIGHT` (142) | 对称/三角矩阵在乘法右侧（B·op(A) 形式）。 |

### aclblasOperation_t

指定对稠密矩阵执行的操作：原矩阵、转置或共轭转置。取值与 Fortran BLAS 中 `N` / `T` / `C` 参数语义一致。

| 取值 | 含义 |
|---|---|
| `ACLBLAS_OP_N` (111) | 不转置，使用原矩阵 op(A) = A。 |
| `ACLBLAS_OP_T` (112) | 转置，op(A) = A<sup>T</sup>。 |
| `ACLBLAS_OP_C` (113) | 共轭转置，op(A) = A<sup>H</sup>；实数矩阵下与 `ACLBLAS_OP_T` 等价。 |

### aclblasComputeType_t

指定 GemmEx / GemmBatchedEx 等扩展 GEMM 接口的内部计算精度。具体支持范围依赖后端实现与输入数据类型。

| 取值 | 含义 |
|---|---|
| `ACLBLAS_COMPUTE_16F` (0) | 计算精度至少为 16 位浮点。 |
| `ACLBLAS_COMPUTE_16F_PEDANTIC` (1) | 计算精度严格为 16 位浮点。 |
| `ACLBLAS_COMPUTE_32F` (2) | 计算精度至少为 32 位浮点。 |
| `ACLBLAS_COMPUTE_32F_PEDANTIC` (3) | 计算精度严格为 32 位浮点。 |
| `ACLBLAS_COMPUTE_32F_FAST_16F` (4) | 32 位输入可降级为 16 位计算以提升性能。 |
| `ACLBLAS_COMPUTE_32F_FAST_16BF` (5) | 32 位输入使用 BF16 路径计算。 |
| `ACLBLAS_COMPUTE_32F_FAST_TF32` (6) | 32 位输入可使用 TF32 等加速计算路径。 |
| `ACLBLAS_COMPUTE_64F` (7) | 计算精度至少为 64 位浮点。 |
| `ACLBLAS_COMPUTE_64F_PEDANTIC` (8) | 计算精度严格为 64 位浮点。 |
| `ACLBLAS_COMPUTE_32I` (9) | 计算精度至少为 32 位整数。 |
| `ACLBLAS_COMPUTE_32I_PEDANTIC` (10) | 计算精度严格为 32 位整数。 |

### aclblasGemmAlgo_t

指定 GemmEx 等接口使用的 GEMM 算法。除默认算法外，其余枚举值为预留，具体支持情况依赖后端。

| 取值 | 含义 |
|---|---|
| `ACLBLAS_GEMM_DEFAULT` (0) | 默认算法，由后端自动选择。 |
| `ACLBLAS_GEMM_ALGO0` (1) ~ `ACLBLAS_GEMM_ALGO7` (8) | 预留算法编号，供后续扩展。 |

### aclblasLapackInfo_t

批量 LAPACK 风格接口（如 `aclblasSgetrfBatched`、`aclblasSgeqrfBatched`）中 `info` / `infoArray` 参数的取值约定，语义对齐 LAPACK `xINFO`：

- `info = 0`：成功退出。
- `info < 0`：若 `info = -i`，表示第 `i` 个参数非法（不含 handle 与 info 本身）。

参数编号遵循 LAPACK 惯例：handle 与 info 不参与计数。例如 `aclblasSgeqrfBatched(handle, m, n, Aarray, lda, TauArray, info, batchSize)` 中，m=1，n=2，Aarray=3，lda=4，TauArray=5，batchSize=6。

| 取值 | 含义 |
|---|---|
| `ACLBLAS_LAPACK_INFO_OK` (0) | 成功退出。 |
| `ACLBLAS_LAPACK_INFO_ARG_1` (-1) | 第 1 个 LAPACK 风格参数非法。 |
| `ACLBLAS_LAPACK_INFO_ARG_2` (-2) | 第 2 个 LAPACK 风格参数非法。 |
| `ACLBLAS_LAPACK_INFO_ARG_3` (-3) | 第 3 个 LAPACK 风格参数非法。 |
| `ACLBLAS_LAPACK_INFO_ARG_4` (-4) | 第 4 个 LAPACK 风格参数非法。 |
| `ACLBLAS_LAPACK_INFO_ARG_5` (-5) | 第 5 个 LAPACK 风格参数非法。 |
| `ACLBLAS_LAPACK_INFO_ARG_6` (-6) | 第 6 个 LAPACK 风格参数非法。 |
| `ACLBLAS_LAPACK_INFO_ARG_7` (-7) | 第 7 个 LAPACK 风格参数非法。 |
| `ACLBLAS_LAPACK_INFO_ARG_8` (-8) | 第 8 个 LAPACK 风格参数非法。 |

## aclBLAS Helper Function Reference

Helper 函数用于库初始化、资源管理、日志配置及 aclBLASLt 描述符管理，不直接执行 BLAS 计算。

### aclBLAS Helper

#### aclblasCreate()

```cpp
aclblasStatus_t aclblasCreate(aclblasHandle_t* handle);
```

初始化 ops-blas 库并在堆上分配 handle，返回 opaque 库上下文。创建时会预分配 32 MiB 默认 workspace（设备内存，由库管理）。调用任何其他 aclBLAS 函数前必须先成功调用本函数。

调用前 `*handle` 必须为 `nullptr`；若 `*handle` 已非空，视为重复创建，返回错误以防止内存泄漏。aclBLAS 库上下文与当前 CANN 设备绑定；多设备场景下每个设备应创建独立 handle。

| 返回值 | 含义 |
|---|---|
| `ACLBLAS_STATUS_SUCCESS` | 创建成功。 |
| `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` | `handle` 为空指针。 |
| `ACLBLAS_STATUS_INVALID_VALUE` | `*handle` 非空。 |
| `ACLBLAS_STATUS_ALLOC_FAILED` | 内存分配失败。 |

#### aclblasDestroy()

```cpp
aclblasStatus_t aclblasDestroy(aclblasHandle_t handle);
```

释放 handle 占用的库内资源。销毁前会同步关联 stream，释放库默认 workspace，并清除用户 workspace 引用（不释放用户自行分配的设备内存）。通常为针对该 handle 的最后一次库调用。

| 返回值 | 含义 |
|---|---|
| `ACLBLAS_STATUS_SUCCESS` | 销毁成功。 |
| `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` | `handle` 为空指针。 |
| `ACLBLAS_STATUS_EXECUTION_FAILED` | stream 同步失败。 |

#### aclblasSetStream()

```cpp
aclblasStatus_t aclblasSetStream(aclblasHandle_t handle, aclrtStream stream);
```

设置 handle 绑定的 AscendCL stream，后续通过该 handle 下发的算子均在此 stream 上执行。若未设置 stream，使用默认 stream。`stream` 为 `nullptr` 时选择默认 stream。

**注意：** 切换 stream 会自动恢复为库默认 workspace；若此前通过 `aclblasSetWorkspace` 设置了用户 workspace，切换 stream 后须重新设置。

| 返回值 | 含义 |
|---|---|
| `ACLBLAS_STATUS_SUCCESS` | 设置成功。 |
| `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` | `handle` 为空指针。 |

#### aclblasGetStream()

```cpp
aclblasStatus_t aclblasGetStream(aclblasHandle_t handle, aclrtStream* stream);
```

获取 handle 当前绑定的 stream。

| 返回值 | 含义 |
|---|---|
| `ACLBLAS_STATUS_SUCCESS` | 获取成功。 |
| `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` | `handle` 为空指针。 |
| `ACLBLAS_STATUS_INVALID_VALUE` | `stream` 为空指针。 |

#### aclblasSetWorkspace()

```cpp
aclblasStatus_t aclblasSetWorkspace(aclblasHandle_t handle, void* workspace, size_t workspaceSize);
```

将用户提供的设备内存借给 handle 作为算子临时存储区，库不取得所有权。支持 grow-only 更新：仅当新 `workspaceSize` 大于当前用户 workspace 大小时才更新设置。**必须同时传入有效的 `workspace` 和 `workspaceSize`**；恢复库默认 workspace 请调用 `aclblasSetStream()`（与 cuBLAS 行为一致）。

算子内部禁止额外 `aclrtMalloc` 分配 workspace；workspace 不足时部分算子可能返回 `ACLBLAS_STATUS_EXECUTION_FAILED`，用户可通过本接口扩容后重试。

| 参数 | 说明 |
|---|---|
| `handle` | aclBLAS 句柄。 |
| `workspace` | 用户分配的设备内存；不可为 `nullptr`。 |
| `workspaceSize` | workspace 字节数；必须大于 0。 |

| 返回值 | 含义 |
|---|---|
| `ACLBLAS_STATUS_SUCCESS` | 设置成功。 |
| `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` | `handle` 为空指针。 |
| `ACLBLAS_STATUS_INVALID_VALUE` | `workspace` 为 `nullptr` 或 `workspaceSize` 为 0。 |

#### aclblasGetVersion()

```cpp
aclblasStatus_t aclblasGetVersion(aclblasHandle_t handle, int* version);
```

返回 ops-blas 库版本号，编码方式为 `MAJOR * 10000 + MINOR * 100 + PATCH`（例如 1.0.0 对应 10000）。`handle` 可为 `nullptr`，允许在不创建 handle 的情况下查询版本。

| 返回值 | 含义 |
|---|---|
| `ACLBLAS_STATUS_SUCCESS` | 查询成功。 |
| `ACLBLAS_STATUS_INVALID_VALUE` | 参数非法（如 `version` 为空）。 |

#### aclblasLoggerConfigure()

```cpp
aclblasStatus_t aclblasLoggerConfigure(
    int logIsOn, int logToStdOut, int logToStdErr, const char* logFile);
```

配置 ops-blas 库运行时日志行为。

| 参数 | 说明 |
|---|---|
| `logIsOn` | 是否开启日志（总开关），非零为开启、0 为关闭。 |
| `logToStdOut` | 是否输出到标准输出，非零为是、0 为否。 |
| `logToStdErr` | 是否输出到标准错误，非零为是、0 为否。 |
| `logFile` | 日志文件路径；为 `nullptr` 时不写文件。调用方需保证该指针在日志输出期间有效（内部不拷贝）。 |

| 返回值 | 含义 |
|---|---|
| `ACLBLAS_STATUS_SUCCESS` | 配置成功。 |

#### aclblasSetLoggerCallback()

```cpp
aclblasStatus_t aclblasSetLoggerCallback(aclblasLogCallback userCallback);
```

安装用户自定义日志回调函数。回调类型为 `void (*)(const char*)`。传入 `nullptr` 可清除已安装的回调。

| 返回值 | 含义 |
|---|---|
| `ACLBLAS_STATUS_SUCCESS` | 设置成功。 |

#### aclblasGetLoggerCallback()

```cpp
aclblasStatus_t aclblasGetLoggerCallback(aclblasLogCallback* userCallback);
```

获取当前已安装的用户自定义日志回调函数指针。

| 返回值 | 含义 |
|---|---|
| `ACLBLAS_STATUS_SUCCESS` | 获取成功。 |
| `ACLBLAS_STATUS_INVALID_VALUE` | `userCallback` 为空指针。 |

### aclBLASLt Helper

#### aclblasLtGetVersion()

```cpp
aclblasStatus_t aclblasLtGetVersion(size_t* version);
```

返回 aclBLASLt 库打包版本号。

| 返回值 | 含义 |
|---|---|
| `ACLBLAS_STATUS_SUCCESS` | 查询成功。 |
| `ACLBLAS_STATUS_INVALID_VALUE` | `version` 为空指针。 |

#### aclblasLtGetProperty()

```cpp
aclblasStatus_t aclblasLtGetProperty(aclblasLtPropertyType_t type, int* value);
```

查询 aclBLASLt 库属性。`type` 取值见 `aclblasLtPropertyType_t`：`ACLBLASLT_PROPERTY_MAJOR_VERSION`、`ACLBLASLT_PROPERTY_MINOR_VERSION`、`ACLBLASLT_PROPERTY_PATCH_LEVEL`。

| 返回值 | 含义 |
|---|---|
| `ACLBLAS_STATUS_SUCCESS` | 查询成功。 |
| `ACLBLAS_STATUS_INVALID_VALUE` | `type` 非法或 `value` 为空指针。 |

#### aclblasLtCreate()

```cpp
aclblasStatus_t aclblasLtCreate(aclblasLtHandle_t* lightHandle);
```

初始化 aclBLASLt 库并创建 opaque 库上下文 handle，分配必要的 host/device 轻量资源。调用任何其他 aclBLASLt 函数前必须先成功调用本函数。库上下文与当前 CANN 设备绑定；多设备场景下每个设备应创建独立 handle。

建议尽量减少 `aclblasLtCreate` / `aclblasLtDestroy` 调用次数，因 Destroy 会隐式触发设备同步。

| 返回值 | 含义 |
|---|---|
| `ACLBLAS_STATUS_SUCCESS` | 创建成功。 |
| `ACLBLAS_STATUS_INVALID_VALUE` | `lightHandle` 为空指针。 |
| `ACLBLAS_STATUS_NOT_INITIALIZED` | CANN 运行环境未初始化。 |
| `ACLBLAS_STATUS_ALLOC_FAILED` | 内存分配失败。 |

#### aclblasLtDestroy()

```cpp
aclblasStatus_t aclblasLtDestroy(const aclblasLtHandle_t lightHandle);
```

释放 aclBLASLt handle 占用的硬件资源，通常为针对该 handle 的最后一次库调用。

| 返回值 | 含义 |
|---|---|
| `ACLBLAS_STATUS_SUCCESS` | 销毁成功。 |
| `ACLBLAS_STATUS_INVALID_VALUE` | `lightHandle` 为空指针或无效句柄。 |

#### aclblasLtMatrixLayoutCreate()

```cpp
aclblasStatus_t aclblasLtMatrixLayoutCreate(aclblasLtMatrixLayout_t* matLayout,
                                            aclDataType type,
                                            uint64_t rows,
                                            uint64_t cols,
                                            int64_t ld);
```

创建矩阵 layout 描述符，用于描述矩阵的数据类型、维度及 leading dimension。列主序下 `ld` 为相邻列之间的元素跨度，须满足 `ld >= rows`。

| 返回值 | 含义 |
|---|---|
| `ACLBLAS_STATUS_SUCCESS` | 创建成功。 |
| `ACLBLAS_STATUS_INVALID_VALUE` | `matLayout` 为空指针或 `ld` 为负数。 |
| `ACLBLAS_STATUS_ALLOC_FAILED` | 描述符内存分配失败。 |

#### aclblasLtMatrixLayoutDestroy()

```cpp
aclblasStatus_t aclblasLtMatrixLayoutDestroy(const aclblasLtMatrixLayout_t matLayout);
```

销毁先前创建的矩阵 layout 描述符。

| 返回值 | 含义 |
|---|---|
| `ACLBLAS_STATUS_SUCCESS` | 销毁成功。 |
| `ACLBLAS_STATUS_INVALID_VALUE` | `matLayout` 为空指针。 |

#### aclblasLtMatrixLayoutSetAttribute()

```cpp
aclblasStatus_t aclblasLtMatrixLayoutSetAttribute(aclblasLtMatrixLayout_t matLayout,
                                                  aclblasLtMatrixLayoutAttribute_t attr,
                                                  const void* buf,
                                                  size_t sizeInBytes);
```

设置矩阵 layout 描述符属性。常用属性包括 batch count、strided batch offset、数据类型、内存 order（`aclblasLtOrder_t`）、行/列数、leading dimension 等，详见 `aclblasLtMatrixLayoutAttribute_t`。

| 返回值 | 含义 |
|---|---|
| `ACLBLAS_STATUS_SUCCESS` | 设置成功。 |
| `ACLBLAS_STATUS_INVALID_VALUE` | `buf` 为空，或 `sizeInBytes` 与属性所需大小不匹配。 |

#### aclblasLtMatrixLayoutGetAttribute()

```cpp
aclblasStatus_t aclblasLtMatrixLayoutGetAttribute(const aclblasLtMatrixLayout_t matLayout,
                                                  aclblasLtMatrixLayoutAttribute_t attr,
                                                  void* buf,
                                                  size_t sizeInBytes,
                                                  size_t* sizeWritten);
```

查询矩阵 layout 描述符属性。`sizeWritten` 可为 `nullptr`；若 `buf` 缓冲区过小，可通过 `sizeWritten` 获知所需大小。

| 返回值 | 含义 |
|---|---|
| `ACLBLAS_STATUS_SUCCESS` | 查询成功。 |
| `ACLBLAS_STATUS_INVALID_VALUE` | 参数为空或缓冲区过小。 |

#### aclblasLtMatmulDescCreate()

```cpp
aclblasStatus_t aclblasLtMatmulDescCreate(aclblasLtMatmulDesc_t* matmulDesc,
                                          aclblasComputeType_t computeType,
                                          aclDataType scaleType);
```

创建矩阵乘法描述符，指定内部计算精度（`computeType`）与缩放因子数据类型（`scaleType`）。后续通过 SetAttribute 配置 transA/transB、epilogue、scale 指针等，供 `aclblasLtMatmul` 使用。

| 返回值 | 含义 |
|---|---|
| `ACLBLAS_STATUS_SUCCESS` | 创建成功。 |
| `ACLBLAS_STATUS_INVALID_VALUE` | `matmulDesc` 为空指针。 |
| `ACLBLAS_STATUS_ALLOC_FAILED` | 描述符内存分配失败。 |

#### aclblasLtMatmulDescDestroy()

```cpp
aclblasStatus_t aclblasLtMatmulDescDestroy(const aclblasLtMatmulDesc_t matmulDesc);
```

销毁先前创建的矩阵乘法描述符。

| 返回值 | 含义 |
|---|---|
| `ACLBLAS_STATUS_SUCCESS` | 销毁成功。 |
| `ACLBLAS_STATUS_INVALID_VALUE` | `matmulDesc` 为空指针。 |

#### aclblasLtMatmulDescInit()

```cpp
aclblasStatus_t aclblasLtMatmulDescInit(aclblasLtMatmulDesc_t matmulDesc,
                                        aclblasComputeType_t computeType,
                                        aclDataType scaleType);
```

初始化调用方预分配的矩阵乘法描述符胶囊（如栈上分配），不分配堆内存。与 `aclblasLtMatmulDescCreate` 的区别：Create 在堆上分配胶囊并返回指针，Init 仅写入已分配的胶囊。通过 Init 初始化的栈胶囊**禁止**传给 `aclblasLtMatmulDescDestroy`（后者会 `delete`）。

| 返回值 | 含义 |
|---|---|
| `ACLBLAS_STATUS_SUCCESS` | 初始化成功。 |
| `ACLBLAS_STATUS_INVALID_VALUE` | `matmulDesc` 为空指针。 |
| `ACLBLAS_STATUS_INTERNAL_ERROR` | 内部拷贝失败。 |

#### aclblasLtMatmulDescSetAttribute()

```cpp
aclblasStatus_t aclblasLtMatmulDescSetAttribute(aclblasLtMatmulDesc_t matmulDesc,
                                                aclblasLtMatmulDescAttribute_t attr,
                                                const void* buf,
                                                size_t sizeInBytes);
```

设置矩阵乘法描述符属性，如 `ACLBLASLT_MATMUL_DESC_TRANSA`、`ACLBLASLT_MATMUL_DESC_TRANSB`、`ACLBLASLT_MATMUL_DESC_EPILOGUE`、各矩阵 scale 指针等，详见 `aclblasLtMatmulDescAttribute_t`。

| 返回值 | 含义 |
|---|---|
| `ACLBLAS_STATUS_SUCCESS` | 设置成功。 |
| `ACLBLAS_STATUS_INVALID_VALUE` | `buf` 为空，或 `sizeInBytes` 与属性所需大小不匹配。 |
| `ACLBLAS_STATUS_NOT_SUPPORTED` | `attr` 不是已识别的属性。 |

#### aclblasLtMatmulDescGetAttribute()

```cpp
aclblasStatus_t aclblasLtMatmulDescGetAttribute(aclblasLtMatmulDesc_t desc,
                                                aclblasLtMatmulDescAttribute_t attr,
                                                void* buf,
                                                size_t sizeInBytes,
                                                size_t* sizeWritten);
```

查询矩阵乘法描述符属性。

| 返回值 | 含义 |
|---|---|
| `ACLBLAS_STATUS_SUCCESS` | 查询成功。 |
| `ACLBLAS_STATUS_INVALID_VALUE` | `desc` 或 `buf` 为空，或缓冲区过小。 |
| `ACLBLAS_STATUS_NOT_SUPPORTED` | `attr` 不是已识别的属性。 |

#### aclblasLtMatmulPreferenceCreate()

```cpp
aclblasStatus_t aclblasLtMatmulPreferenceCreate(aclblasLtMatmulPreference_t* pref);
```

创建矩阵乘法算法搜索偏好描述符，用于 `aclblasLtMatmulAlgoGetHeuristic` 约束 workspace 上限等搜索条件。

| 返回值 | 含义 |
|---|---|
| `ACLBLAS_STATUS_SUCCESS` | 创建成功。 |
| `ACLBLAS_STATUS_INVALID_VALUE` | `pref` 为空指针。 |
| `ACLBLAS_STATUS_ALLOC_FAILED` | 描述符内存分配失败。 |

#### aclblasLtMatmulPreferenceDestroy()

```cpp
aclblasStatus_t aclblasLtMatmulPreferenceDestroy(const aclblasLtMatmulPreference_t pref);
```

销毁先前创建的算法搜索偏好描述符。

| 返回值 | 含义 |
|---|---|
| `ACLBLAS_STATUS_SUCCESS` | 销毁成功。 |
| `ACLBLAS_STATUS_INVALID_VALUE` | `pref` 为空指针。 |

#### aclblasLtMatmulPreferenceSetAttribute()

```cpp
aclblasStatus_t aclblasLtMatmulPreferenceSetAttribute(aclblasLtMatmulPreference_t pref,
                                                      aclblasLtMatmulPreferenceAttribute_t attr,
                                                      const void* buf,
                                                      size_t sizeInBytes);
```

设置算法搜索偏好属性，如 `ACLBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES`（允许的最大 workspace 字节数，默认 32MB）。

| 返回值 | 含义 |
|---|---|
| `ACLBLAS_STATUS_SUCCESS` | 设置成功。 |
| `ACLBLAS_STATUS_INVALID_VALUE` | `buf` 为空，或 `sizeInBytes` 与属性所需大小不匹配。 |
| `ACLBLAS_STATUS_NOT_SUPPORTED` | `attr` 不是已识别的属性。 |

#### aclblasLtMatmulPreferenceGetAttribute()

```cpp
aclblasStatus_t aclblasLtMatmulPreferenceGetAttribute(aclblasLtMatmulPreference_t pref,
                                                      aclblasLtMatmulPreferenceAttribute_t attr,
                                                      void* buf,
                                                      size_t sizeInBytes,
                                                      size_t* sizeWritten);
```

查询算法搜索偏好描述符属性。

| 返回值 | 含义 |
|---|---|
| `ACLBLAS_STATUS_SUCCESS` | 查询成功。 |
| `ACLBLAS_STATUS_INVALID_VALUE` | `pref` 或 `buf` 为空，或缓冲区过小。 |
| `ACLBLAS_STATUS_NOT_SUPPORTED` | `attr` 不是已识别的属性。 |

#### aclblasLtMatmulAlgoInit()

```cpp
aclblasStatus_t aclblasLtMatmulAlgoInit(aclblasLtHandle_t lightHandle,
                                        aclblasComputeType_t computeType,
                                        aclDataType scaleType,
                                        aclDataType Atype,
                                        aclDataType Btype,
                                        aclDataType Ctype,
                                        aclDataType Dtype,
                                        int algoId,
                                        aclblasLtMatmulAlgo_t* algo);
```

根据算法 ID 和数据类型组合初始化 `aclblasLtMatmulAlgo_t` 结构体。`algoId=0` 请求默认配置（编码后为 `0x00808201`）；非 0 值须为合法编码 ID，否则返回 `INVALID_VALUE`。

| 返回值 | 含义 |
|---|---|
| `ACLBLAS_STATUS_SUCCESS` | 初始化成功。 |
| `ACLBLAS_STATUS_NOT_INITIALIZED` | `lightHandle` 为空指针。 |
| `ACLBLAS_STATUS_INVALID_VALUE` | `algo` 为空、`algoId` 为负数或非零 `algoId` 无法解码。 |
| `ACLBLAS_STATUS_NOT_SUPPORTED` | 数据类型组合不支持。 |

#### aclblasLtMatmulAlgoConfigSetAttribute()

```cpp
aclblasStatus_t aclblasLtMatmulAlgoConfigSetAttribute(aclblasLtMatmulAlgo_t* algo,
                                                      aclblasLtMatmulAlgoConfigAttributes_t attr,
                                                      const void* buf,
                                                      size_t sizeInBytes);
```

设置算法描述符的配置属性。属性到内部字段的映射详见 `aclblasLtMatmulAlgoConfigAttributes_t`。每次 Set 后自动重算 `algoId`，其中 l1k 取自 algo 内部 flags 高位编码的 K 维 tile（由 Heuristic/AlgoInit 设置，默认 128），不会因修改其他属性而被重置。

| 返回值 | 含义 |
|---|---|
| `ACLBLAS_STATUS_SUCCESS` | 属性设置成功。 |
| `ACLBLAS_STATUS_INVALID_VALUE` | 指针为空、`sizeInBytes` 不匹配、值超范围、tileId 非法或 magic 校验失败。 |
| `ACLBLAS_STATUS_NOT_SUPPORTED` | 属性为只读（`ALGO_CONFIG_ID`）、不支持（`INNER_SHAPE_ID`）或未识别。 |

#### aclblasLtMatmulAlgoConfigGetAttribute()

```cpp
aclblasStatus_t aclblasLtMatmulAlgoConfigGetAttribute(const aclblasLtMatmulAlgo_t* algo,
                                                      aclblasLtMatmulAlgoConfigAttributes_t attr,
                                                      void* buf,
                                                      size_t sizeInBytes,
                                                      size_t* sizeWritten);
```

查询算法描述符的配置属性。当 `sizeInBytes=0` 时，仅通过 `sizeWritten` 返回所需字节数，不写 `buf`（对齐 cuBLAS 探长语义）。

| 返回值 | 含义 |
|---|---|
| `ACLBLAS_STATUS_SUCCESS` | 查询成功（含 `sizeInBytes=0` 探长）。 |
| `ACLBLAS_STATUS_INVALID_VALUE` | 指针为空、缓冲区过小或 magic 校验失败。 |
| `ACLBLAS_STATUS_NOT_SUPPORTED` | 属性未识别或不支持。 |

#### aclblasLtMatmulAlgoGetIds()

```cpp
aclblasStatus_t aclblasLtMatmulAlgoGetIds(aclblasLtHandle_t lightHandle,
                                          aclblasComputeType_t computeType,
                                          aclDataType scaleType,
                                          aclDataType Atype,
                                          aclDataType Btype,
                                          aclDataType Ctype,
                                          aclDataType Dtype,
                                          int* algoIdsArray,
                                          int algoIdsArrayLength,
                                          int* numAlgoIds);
```

查询指定计算类型与数据类型组合下 aclBLASLt 支持的全部算法 ID 集合，每个 ID 均可作为 `aclblasLtMatmulAlgoInit` 的合法 `algoId` 输入。ID 按升序写入 `algoIdsArray`；当 `algoIdsArrayLength` 小于可用 ID 数量时，仅写入前 `algoIdsArrayLength` 个，实际写入数量通过 `numAlgoIds` 返回。

| 返回值 | 含义 |
|---|---|
| `ACLBLAS_STATUS_SUCCESS` | 查询成功。 |
| `ACLBLAS_STATUS_NOT_INITIALIZED` | `lightHandle` 为空指针。 |
| `ACLBLAS_STATUS_INVALID_VALUE` | `algoIdsArray` 或 `numAlgoIds` 为空指针，或 `algoIdsArrayLength <= 0`。 |
| `ACLBLAS_STATUS_NOT_SUPPORTED` | 数据类型组合不支持。 |

#### aclblasLtMatmulAlgoGetHeuristic()

```cpp
aclblasStatus_t aclblasLtMatmulAlgoGetHeuristic(aclblasLtHandle_t lightHandle,
                                                aclblasLtMatmulDesc_t matmulDesc,
                                                aclblasLtMatrixLayout_t Adesc,
                                                aclblasLtMatrixLayout_t Bdesc,
                                                aclblasLtMatrixLayout_t Cdesc,
                                                aclblasLtMatrixLayout_t Ddesc,
                                                aclblasLtMatmulPreference_t pref,
                                                int requestedAlgoCount,
                                                aclblasLtMatmulHeuristicResult_t heuristicResultsArray[],
                                                int* returnAlgoCount);
```

根据给定的 matmul 描述符与 A/B/C/D 矩阵 layout，检索可用的矩阵乘法算法启发式结果。结果按预估计算时间递增顺序写入 `heuristicResultsArray`；实际返回数量写入 `returnAlgoCount`。调用 `aclblasLtMatmul` 时可将 `heuristicResultsArray[i].algo` 传入 `algo` 参数。

成功返回后应检查 `heuristicResultsArray[0 .. returnAlgoCount-1].state` 确认各候选算法是否可用。

| 返回值 | 含义 |
|---|---|
| `ACLBLAS_STATUS_SUCCESS` | 查询成功（须进一步检查各结果项的 `state`）。 |
| `ACLBLAS_STATUS_INVALID_VALUE` | 任一指针参数为空、`requestedAlgoCount` 小于等于 0，或计算类型与输入矩阵数据类型不兼容。 |

#### aclblasLtMatrixTransformDescCreate()

```cpp
aclblasStatus_t aclblasLtMatrixTransformDescCreate(aclblasLtMatrixTransformDesc_t* transformDesc,
                                                   aclDataType scaleType);
```

创建矩阵变换描述符，指定计算（scale）精度类型，供 `aclblasLtMatrixTransform` 使用。

| 返回值 | 含义 |
|---|---|
| `ACLBLAS_STATUS_SUCCESS` | 创建成功。 |
| `ACLBLAS_STATUS_INVALID_VALUE` | `transformDesc` 为空指针。 |
| `ACLBLAS_STATUS_ALLOC_FAILED` | 描述符内存分配失败。 |

#### aclblasLtMatrixTransformDescDestroy()

```cpp
aclblasStatus_t aclblasLtMatrixTransformDescDestroy(const aclblasLtMatrixTransformDesc_t transformDesc);
```

销毁先前创建的矩阵变换描述符。

| 返回值 | 含义 |
|---|---|
| `ACLBLAS_STATUS_SUCCESS` | 销毁成功。 |
| `ACLBLAS_STATUS_INVALID_VALUE` | `transformDesc` 为空指针。 |

#### aclblasLtMatrixTransformDescSetAttribute()

```cpp
aclblasStatus_t aclblasLtMatrixTransformDescSetAttribute(aclblasLtMatrixTransformDesc_t transformDesc,
                                                         aclblasLtMatrixTransformDescAttribute_t attr,
                                                         const void* buf,
                                                         size_t sizeInBytes);
```

设置矩阵变换描述符属性，如 `TRANSA`/`TRANSB`（`aclblasOperation_t`）、pointer mode 等，详见 `aclblasLtMatrixTransformDescAttribute_t`。

| 返回值 | 含义 |
|---|---|
| `ACLBLAS_STATUS_SUCCESS` | 设置成功。 |
| `ACLBLAS_STATUS_INVALID_VALUE` | 参数为空或 `sizeInBytes` 不匹配。 |
| `ACLBLAS_STATUS_NOT_SUPPORTED` | `attr` 不是已识别的属性。 |

#### aclblasLtMatrixTransformDescGetAttribute()

```cpp
aclblasStatus_t aclblasLtMatrixTransformDescGetAttribute(aclblasLtMatrixTransformDesc_t transformDesc,
                                                         aclblasLtMatrixTransformDescAttribute_t attr,
                                                         void* buf,
                                                         size_t sizeInBytes,
                                                         size_t* sizeWritten);
```

查询矩阵变换描述符属性。

| 返回值 | 含义 |
|---|---|
| `ACLBLAS_STATUS_SUCCESS` | 查询成功。 |
| `ACLBLAS_STATUS_INVALID_VALUE` | 参数为空或缓冲区过小。 |
| `ACLBLAS_STATUS_NOT_SUPPORTED` | `attr` 不是已识别的属性。 |

#### aclblasLtLoggerSetFile()

```cpp
aclblasStatus_t aclblasLtLoggerSetFile(FILE* file);
```

设置日志输出文件句柄。注册后该文件句柄不得关闭，除非再次调用本函数切换到其他文件句柄。若从未调用本函数，日志默认输出到 stdout。

| 参数 | 说明 |
|---|---|
| `file` | 已打开且具有写权限的文件指针。 |

| 返回值 | 含义 |
|---|---|
| `ACLBLAS_STATUS_SUCCESS` | 设置成功。 |
| `ACLBLAS_STATUS_INVALID_VALUE` | `file` 为空指针。 |

#### aclblasLtLoggerSetLevel()

```cpp
aclblasStatus_t aclblasLtLoggerSetLevel(int level);
```

设置日志级别。仅在掩码允许且级别匹配时输出日志消息。取值见 `aclblasLtLogLevel_t`，有效范围为 [0, 5]。

| 参数 | 说明 |
|---|---|
| `level` | 日志级别，取值见 `aclblasLtLogLevel_t`。 |

| 返回值 | 含义 |
|---|---|
| `ACLBLAS_STATUS_SUCCESS` | 设置成功。 |
| `ACLBLAS_STATUS_INVALID_VALUE` | `level` 不是合法日志级别。 |

#### aclblasLtLoggerSetMask()

```cpp
aclblasStatus_t aclblasLtLoggerSetMask(int mask);
```

设置日志掩码，以位掩码形式控制各类消息的输出。取值为 `aclblasLtLogMask_t` 标志的按位 OR 组合（例如 `5` = Error + Hints）。

| 参数 | 说明 |
|---|---|
| `mask` | 日志掩码（`ACLBLASLT_LOG_MASK_*` 的按位 OR）。 |

| 返回值 | 含义 |
|---|---|
| `ACLBLAS_STATUS_SUCCESS` | 设置成功。 |

#### aclblasLtLoggerForceDisable()

```cpp
aclblasStatus_t aclblasLtLoggerForceDisable(void);
```

强制禁用日志。调用后整个进程生命周期内无法通过任何其他 Logger API 或环境变量重新启用日志。

| 返回值 | 含义 |
|---|---|
| `ACLBLAS_STATUS_SUCCESS` | 已成功禁用日志。 |

#### aclblasLtLoggerSetCallback()

```cpp
aclblasStatus_t aclblasLtLoggerSetCallback(aclblasLtLoggerCallback_t callback);
```

设置日志回调函数。设置后日志消息将传递给回调函数，而非写入文件或 stdout。传入 `nullptr` 可清除已设置的回调。

| 参数 | 说明 |
|---|---|
| `callback` | 回调函数指针，见 `aclblasLtLoggerCallback_t`。传 `nullptr` 清除回调。 |

| 返回值 | 含义 |
|---|---|
| `ACLBLAS_STATUS_SUCCESS` | 设置成功。 |

#### aclblasLtLoggerOpenFile()

```cpp
aclblasStatus_t aclblasLtLoggerOpenFile(const char* logFile);
```

按路径打开日志文件并设为日志输出目标。文件句柄由库内部管理，在重新配置日志输出或程序退出时自动关闭。

| 参数 | 说明 |
|---|---|
| `logFile` | 日志文件路径。 |

| 返回值 | 含义 |
|---|---|
| `ACLBLAS_STATUS_SUCCESS` | 文件打开成功。 |
| `ACLBLAS_STATUS_INVALID_VALUE` | `logFile` 为空指针。 |
| `ACLBLAS_STATUS_INTERNAL_ERROR` | 文件无法打开。 |

## aclBLAS Level 1 Function Reference

Level 1 接口在向量之间进行运算，典型操作包括向量缩放（scal）、向量加法（axpy）、点积（dot）、范数（nrm2）、元素交换（swap）等。

算子接口列表如下：

| 接口名 | 说明 |
|---|---|
| [aclblasSasum](../../blas/asum/README.md) | 实数向量绝对值之和 |
| [aclblasScasum](../../blas/asum/README.md) | 复数（complex64）向量绝对值分量之和（Ascend 950PR 支持） |
| [aclblasSaxpy](../../blas/axpy/README.md) | 单精度浮点 AXPY（y = αx + y） |
| [aclblasCaxpy](../../blas/axpy/README.md) | 复数 AXPY |
| [aclblasCdgmm](../../blas/dgmm/README.md) | 行主序单精度复数对角矩阵左乘 |
| [aclblasComplexMatDot](../../extensions/complexmatdot/README.md) | 复数矩阵逐元素点乘 |
| [aclblasScopy](../../blas/copy/README.md) | 单精度浮点向量拷贝 |
| [aclblasCcopy](../../blas/copy/README.md) | 复数向量拷贝 |
| [aclblasSdot](../../blas/dot/README.md) | 实数向量点积 |
| [aclblasCdotu](../../blas/dot/README.md) | 无共轭复数点积 |
| [aclblasCdotc](../../blas/dot/README.md) | 共轭复数点积 |
| [aclblasIsamax](../../blas/iamax/README.md) | 查找向量中绝对值最大元素的索引 |
| [aclblasSnrm2](../../blas/nrm2/README.md) | 实数向量欧几里得范数 |
| [aclblasScnrm2](../../blas/nrm2/README.md) | 复数向量欧几里得范数 |
| [aclblasCsrot](../../blas/rot/README.md) | 复数向量平面旋转 |
| [aclblasSrotm](../../blas/rotm/README.md) | 实数向量 Modified Givens 旋转 |
| [aclblasSrotmg](../../blas/rotmg/README.md) | 构造修正 Givens 旋转参数 |
| [aclblasSscal](../../blas/scal/README.md) | 实数向量乘以标量 |
| [aclblasCscal](../../blas/scal/README.md) | 复数向量乘以复数标量 |
| [aclblasCsscal](../../blas/scal/README.md) | 复数向量乘以实数标量 |
| [aclblasScalex](../../blas/scalex/README.md) | 混合精度向量标量乘 |
| [aclblasSswap](../../blas/swap/README.md) | 实数向量交换 |
| [aclblasCswap](../../blas/swap/README.md) | 复数向量交换 |

## aclBLAS Level 2 Function Reference

Level 2 接口在矩阵与向量之间进行运算，典型操作包括通用矩阵-向量乘法（gemv）、对称矩阵-向量乘法（symv）、秩-1 更新（ger）、三角矩阵-向量乘法（trmv）等。

算子接口列表如下：

| 接口名 | 说明 |
|---|---|
| [aclblasSgbmv](../../blas/gbmv/README.md) | 单精度浮点带状矩阵-向量乘法 |
| [aclblasCgbmv](../../blas/gbmv/README.md) | 单精度复数带状矩阵-向量乘法 |
| [aclblasSgemv](../../blas/gemv/README.md) | 单精度浮点矩阵-向量乘法 |
| [aclblasCgemv](../../blas/gemv/README.md) | 复数矩阵-向量乘法 |
| [aclblasSgemvBatched](../../blas/gemv_batched/README.md) | 单精度批量矩阵-向量乘法 |
| [aclblasHSHgemvBatched](../../blas/gemv_batched/README.md) | FP16 入/出批量矩阵-向量乘法 |
| [aclblasHSSgemvBatched](../../blas/gemv_batched/README.md) | FP16 入/FP32 出批量矩阵-向量乘法 |
| [aclblasTSTgemvBatched](../../blas/gemv_batched/README.md) | FP16 入/FP16 出批量矩阵-向量乘法（T 精度变体） |
| [aclblasTSSgemvBatched](../../blas/gemv_batched/README.md) | FP16 入/FP32 出批量矩阵-向量乘法（T 精度变体） |
| [aclblasCgemvBatched](../../blas/gemv_batched/README.md) | 复数批量矩阵-向量乘法 |
| [aclblasSger](../../blas/ger/README.md) | 单精度浮点矩阵秩-1 更新 |
| [aclblasCgerc](../../blas/gerc/README.md) | 复数矩阵共轭秩-1 更新 |
| [aclblasSsbmv](../../blas/sbmv/README.md) | 单精度浮点对称带状矩阵-向量乘法 |
| [aclblasSspmv](../../blas/spmv/README.md) | 单精度浮点对称 packed 矩阵-向量乘法 |
| [aclblasSspr](../../blas/spr/README.md) | 单精度对称 packed 秩-1 更新 |
| [aclblasChpr](../../blas/hpr/README.md) | 单精度复数 Hermitian packed 秩-1 更新 |
| [aclblasSspr2](../../blas/spr2/README.md) | 单精度对称矩阵 packed 格式秩-2 更新 |
| [aclblasSsymv](../../blas/symv/README.md) | 单精度对称矩阵-向量乘法 |
| [aclblasCsymv](../../blas/symv/README.md) | 单精度复数对称矩阵-向量乘法 |
| [aclblasStbmv](../../blas/tbmv/README.md) | 单精度三角带状矩阵-向量乘法（标准接口） |
| [aclblasStpmv](../../blas/tpmv/README.md) | 单精度三角压缩矩阵-向量乘法（标准接口） |
| [aclblasStpsv](../../blas/tpsv/README.md) | 单精度三角 packed 矩阵求解 |
| [aclblasCtpsv](../../blas/tpsv/README.md) | 单精度复数三角 packed 矩阵求解 |
| [aclblasStpttr](../../blas/tpttr/README.md) | 单精度压缩三角矩阵展开为常规矩阵 |
| [aclblasStrmv](../../blas/trmv/README.md) | 实数三角矩阵-向量乘法 |
| [aclblasCtrmv](../../blas/trmv/README.md) | 复数三角矩阵-向量乘法 |
| [aclblasStrsv](../../blas/trsv/README.md) | 单精度三角矩阵求解 |
| [aclblasStrttp](../../blas/trttp/README.md) | 单精度常规三角矩阵压缩为 packed 格式 |
| [aclblasSsyr](../../blas/syr/README.md) | 单精度对称秩-1 更新 |
| [aclblasSsyr2](../../blas/syr2/README.md) | 单精度对称秩-2 更新 |

## aclBLAS Level 3 Function Reference

Level 3 接口在矩阵之间进行运算，典型操作包括对称矩阵乘法（symm）、分组批量 GEMM、批量 LAPACK 分解与求解等。

算子接口列表如下：

| 接口名 | 说明 |
|---|---|
| [aclblasSgemm](../../blas/gemm/README.md) | 单精度浮点通用矩阵乘法 |
| [aclblasCgemm](../../blas/gemm/README.md) | 复数通用矩阵乘法 |
| [aclblasSgemmGroupedBatched](../../blas/gemm_grouped_batched/README.md) | 单精度浮点分组批量矩阵乘法 |
| [aclblasSgelsBatched](../../blas/gels_batched/README.md) | 单精度批量最小二乘/最小范数求解 |
| [aclblasSgeqrfBatched](../../blas/geqrf_batched/README.md) | 单精度批量 QR 分解 |
| [aclblasSgetrfBatched](../../blas/getrf_batched/README.md) | 单精度批量 LU 分解（带部分主元选取） |
| [aclblasSgetriBatched](../../blas/getri_batched/README.md) | 单精度批量矩阵求逆 |
| [aclblasSgetrsBatched](../../blas/getrs_batched/README.md) | 单精度批量线性方程组求解 |
| [aclblasSmatinvBatched](../../blas/matinv_batched/README.md) | 单精度批量矩阵求逆 |
| [aclblasSsymm](../../blas/symm/README.md) | 单精度浮点对称矩阵乘法 |
| [aclblasCtrsm](../../blas/trsm/README.md#aclblasctrsm) | 单精度复数三角矩阵线性方程组求解（Ascend 950PR） |

## BLAS-like Extension

BLAS-like Extension 提供标准 BLAS Level 3 之外的扩展 GEMM 接口（以 **Ex** 为后缀），通过 `aclblasComputeType_t`、`aclblasGemmAlgo_t` 指定计算精度与算法。适用于混合精度、量化等场景。

算子接口列表如下：

| 接口名 | 说明 |
|---|---|
| [aclblasSgeam](../../blas/geam/README.md) | 单精度矩阵加法：C = alpha * op(A) + beta * op(B) |
| [aclblasCgeam](../../blas/geam/README.md) | 复数矩阵加法：C = alpha * op(A) + beta * op(B) |
| [aclblasGemmEx](../../blas/gemm_ex/README.md) | 通用矩阵乘法扩展接口，支持 A/B/C 独立数据类型 |
| [aclblasGemmBatchedEx](../../blas/gemm_batched_ex/README.md) | 通用矩阵乘法批量扩展接口 |
| [aclblasGemmStridedBatchedEx](../../blas/gemm_strided_batched_ex/README.md) | 通用矩阵乘法跨步批量扩展接口 |
| [aclblasGemmGroupedBatchedEx](../../blas/gemm_grouped_batched_ex/README.md) | 通用矩阵乘法分组批量扩展接口 |

## aclBLASLt Datatypes Reference

### aclblasLtLoggerCallback_t

日志回调函数指针类型。通过 `aclblasLtLoggerSetCallback` 设置后，库内日志消息将通过该回调输出，替代文件或 stdout 输出。

```c
typedef void (*aclblasLtLoggerCallback_t)(int logLevel,
                                          const char* functionName,
                                          const char* message);
```

| 参数 | 说明 |
|---|---|
| `logLevel` | 日志消息的级别，见 `aclblasLtLogLevel_t`。 |
| `functionName` | 产生该日志的 API 函数名。 |
| `message` | 日志消息文本（以 null 结尾）。 |

### aclblasLtLogLevel_t

aclBLASLt 日志级别枚举。与 cuBLASLt 的 `CUBLASLT_LOG_LEVEL` 取值一一对应。

| 取值 | 含义 |
|---|---|
| `ACLBLASLT_LOG_LEVEL_OFF` (0) | 禁用日志（默认）。 |
| `ACLBLASLT_LOG_LEVEL_ERROR` (1) | 仅记录错误。 |
| `ACLBLASLT_LOG_LEVEL_TRACE` (2) | 记录启动 kernel 的 API 调用参数与关键信息。 |
| `ACLBLASLT_LOG_LEVEL_HINTS` (3) | 记录可能提升性能的提示。 |
| `ACLBLASLT_LOG_LEVEL_INFO` (4) | 记录库执行的通用信息。 |
| `ACLBLASLT_LOG_LEVEL_API_TRACE` (5) | 记录所有 API 调用的参数与关键信息。 |

### aclblasLtLogMask_t

aclBLASLt 日志掩码位标志枚举，按位组合控制各类消息的输出。与 cuBLASLt 的 `CUBLASLT_LOG_MASK` 取值一一对应。

| 取值 | 含义 |
|---|---|
| `ACLBLASLT_LOG_MASK_OFF` (0) | 关闭。 |
| `ACLBLASLT_LOG_MASK_ERROR` (1) | 错误消息。 |
| `ACLBLASLT_LOG_MASK_TRACE` (2) | kernel 启动追踪。 |
| `ACLBLASLT_LOG_MASK_HINTS` (4) | 性能提示。 |
| `ACLBLASLT_LOG_MASK_INFO` (8) | 通用信息。 |
| `ACLBLASLT_LOG_MASK_API_TRACE` (16) | API 追踪。 |

### aclblasLtOrder_t

指定矩阵的内存布局 order。用于 `ACLBLASLT_MATRIX_LAYOUT_ORDER` 属性。

| 取值 | 含义 |
|---|---|
| `ACLBLASLT_ORDER_COL` (0) | 列主序（column major）。 |
| `ACLBLASLT_ORDER_ROW` (1) | 行主序（row major）。 |
| `ACLBLASLT_ORDER_COL32` (2) | 32 列复合分块，分块内列主序。 |
| `ACLBLASLT_ORDER_COL4_4R2_8C` (3) | 32 列 8 行复合量化分块。 |
| `ACLBLASLT_ORDER_COL32_2R_4R4` (4) | 32 列 32 行复合量化分块。 |

### aclblasLtPropertyType_t

`aclblasLtGetProperty` 查询的库属性类型。

| 取值 | 含义 |
|---|---|
| `ACLBLASLT_PROPERTY_MAJOR_VERSION` (0) | 主版本号。 |
| `ACLBLASLT_PROPERTY_MINOR_VERSION` (1) | 次版本号。 |
| `ACLBLASLT_PROPERTY_PATCH_LEVEL` (2) | 补丁版本号。 |

### aclblasLtEpilogue_t

指定矩阵乘法结果的后处理（epilogue）融合操作，通过 `ACLBLASLT_MATMUL_DESC_EPILOGUE` 属性设置。取值为位掩码，可组合（例如 `BIAS | RELU` = 6）。`AUX` 类变体会将 GEMM 原始结果写入辅助缓冲区。

| 取值 | 含义 |
|---|---|
| `ACLBLASLT_EPILOGUE_DEFAULT` (1) | 无特殊后处理；必要时执行 scale / 量化。 |
| `ACLBLASLT_EPILOGUE_RELU` (2) | 对结果逐点应用 ReLU（`x := max(x, 0)`）。 |
| `ACLBLASLT_EPILOGUE_BIAS` (4) | 广播偏置向量并相加；偏置长度须等于 D 的行数且 stride 为 1。 |
| `ACLBLASLT_EPILOGUE_RELU_BIAS` (6) | 先加偏置，再应用 ReLU。 |
| `ACLBLASLT_EPILOGUE_GELU` (32) | 对结果逐点应用 GELU（`x := GELU(x)`）。 |
| `ACLBLASLT_EPILOGUE_GELU_BIAS` (36) | 先加偏置，再应用 GELU。 |
| `ACLBLASLT_EPILOGUE_RELU_AUX` (130) | 输出 GEMM 原始结果至辅助缓冲区，再应用 ReLU。 |
| `ACLBLASLT_EPILOGUE_RELU_AUX_BIAS` (134) | 输出加偏置后的 GEMM 结果至辅助缓冲区，再应用 ReLU。 |
| `ACLBLASLT_EPILOGUE_DRELU` (136) | 应用 ReLU 梯度变换，需额外辅助输入。 |
| `ACLBLASLT_EPILOGUE_DRELU_BGRAD` (152) | 应用 ReLU 梯度变换并对偏置求梯度，需额外辅助输入。 |
| `ACLBLASLT_EPILOGUE_GELU_AUX` (160) | 输出 GEMM 原始结果至辅助缓冲区，再应用 GELU。 |
| `ACLBLASLT_EPILOGUE_GELU_AUX_BIAS` (164) | 输出加偏置后的 GEMM 结果至辅助缓冲区，再应用 GELU。 |
| `ACLBLASLT_EPILOGUE_DGELU` (192) | 应用 GELU 梯度变换，需额外辅助输入。 |
| `ACLBLASLT_EPILOGUE_DGELU_BGRAD` (208) | 应用 GELU 梯度变换并对偏置求梯度，需额外辅助输入。 |
| `ACLBLASLT_EPILOGUE_BGRADA` (256) | 对 A 求偏置梯度并输出 GEMM 结果。 |
| `ACLBLASLT_EPILOGUE_BGRADB` (512) | 对 B 求偏置梯度并输出 GEMM 结果。 |
| `ACLBLASLT_EPILOGUE_SIGMOID` (1024) | 对结果逐点应用 sigmoid 激活函数。 |
| `ACLBLASLT_EPILOGUE_SWISH_EXT` (65536) | 对结果逐点应用 Swish（`x := Swish(x, 1)`）。 |
| `ACLBLASLT_EPILOGUE_SWISH_BIAS_EXT` (65540) | 先加偏置，再应用 Swish。 |
| `ACLBLASLT_EPILOGUE_CLAMP_EXT` (131072) | 对结果逐点 clamp（`x := max(alpha, min(x, beta))`）。 |
| `ACLBLASLT_EPILOGUE_CLAMP_BIAS_EXT` (131076) | 先加偏置，再 clamp。 |
| `ACLBLASLT_EPILOGUE_CLAMP_AUX_EXT` (131200) | 输出 GEMM 原始结果至辅助缓冲区，再 clamp。 |
| `ACLBLASLT_EPILOGUE_CLAMP_AUX_BIAS_EXT` (131204) | 输出加偏置后的 GEMM 结果至辅助缓冲区，再 clamp。 |

### aclblasLtMatrixLayoutAttribute_t

矩阵 layout 描述符的可配置属性，用于 `aclblasLtMatrixLayoutSetAttribute` / `aclblasLtMatrixLayoutGetAttribute`。

| 取值 | 含义 | 数据类型 | 默认值 |
|---|---|---|---|
| `ACLBLASLT_MATRIX_LAYOUT_BATCH_COUNT` (0) | batch 数量。 | `int32_t` | 1 |
| `ACLBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET` (1) | strided-batch 偏移（元素数）。 | `int64_t` | 0 |
| `ACLBLASLT_MATRIX_LAYOUT_TYPE` (2) | 矩阵数据类型，见 `aclDataType`。 | `uint32_t` | 创建时指定 |
| `ACLBLASLT_MATRIX_LAYOUT_ORDER` (3) | 内存 order，见 `aclblasLtOrder_t`。 | `int32_t` | `ACLBLASLT_ORDER_COL` |
| `ACLBLASLT_MATRIX_LAYOUT_ROWS` (4) | 行数。 | `uint64_t` | 创建时指定 |
| `ACLBLASLT_MATRIX_LAYOUT_COLS` (5) | 列数。 | `uint64_t` | 创建时指定 |
| `ACLBLASLT_MATRIX_LAYOUT_LD` (6) | leading dimension（元素数）。 | `int64_t` | 创建时指定 |

### aclblasLtMatmulDescAttribute_t

矩阵乘法描述符的可配置属性，用于 `aclblasLtMatmulDescSetAttribute` / `aclblasLtMatmulDescGetAttribute`。

| 取值 | 含义 | 数据类型 | 默认值 |
|---|---|---|---|
| `ACLBLASLT_MATMUL_DESC_TRANSA` (0) | 对矩阵 A 的变换操作，见 `aclblasOperation_t`。 | `int32_t` | `ACLBLAS_OP_N` |
| `ACLBLASLT_MATMUL_DESC_TRANSB` (1) | 对矩阵 B 的变换操作，见 `aclblasOperation_t`。 | `int32_t` | `ACLBLAS_OP_N` |
| `ACLBLASLT_MATMUL_DESC_EPILOGUE` (2) | epilogue 后处理，见 `aclblasLtEpilogue_t`。 | `uint32_t` | `ACLBLASLT_EPILOGUE_DEFAULT` |
| `ACLBLASLT_MATMUL_DESC_BIAS_POINTER` (3) | 设备侧偏置 / 偏置梯度向量指针。 | `void*` / `const void*` | NULL |
| `ACLBLASLT_MATMUL_DESC_BIAS_DATA_TYPE` (4) | 偏置向量数据类型，可同 D 矩阵或 scale 类型。 | `int32_t`（`aclDataType`） | — |
| `ACLBLASLT_MATMUL_DESC_A_SCALE_POINTER` (5) | A 的 scale 因子设备指针；为 NULL 时视为 1。 | `void*` / `const void*` | NULL |
| `ACLBLASLT_MATMUL_DESC_B_SCALE_POINTER` (6) | 同 A_SCALE_POINTER，作用于矩阵 B。 | `void*` / `const void*` | NULL |
| `ACLBLASLT_MATMUL_DESC_C_SCALE_POINTER` (7) | 同 A_SCALE_POINTER，作用于矩阵 C。 | `void*` / `const void*` | NULL |
| `ACLBLASLT_MATMUL_DESC_D_SCALE_POINTER` (8) | 同 A_SCALE_POINTER，作用于矩阵 D。 | `void*` / `const void*` | NULL |
| `ACLBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER` (9) | 同 A_SCALE_POINTER，作用于辅助缓冲区。 | `void*` / `const void*` | NULL |
| `ACLBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER` (10) | epilogue 辅助缓冲区设备指针。 | `void*` / `const void*` | NULL |
| `ACLBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD` (11) | 辅助缓冲区的 leading dimension。 | `int64_t` | — |
| `ACLBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE` (12) | 辅助缓冲区的 batch stride。 | `int64_t` | — |
| `ACLBLASLT_MATMUL_DESC_POINTER_MODE` (13) | alpha / beta 的传递方式（host / device / 向量）。 | `int32_t` | host |
| `ACLBLASLT_MATMUL_DESC_AMAX_D_POINTER` (14) | 完成时写入 D 矩阵绝对值最大值的设备指针。 | `void*` / `const void*` | NULL |
| `ACLBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE` (22) | 辅助向量数据类型；默认使用 D 矩阵类型。 | `int32_t`（`aclDataType`） | D 矩阵类型 |
| `ACLBLASLT_MATMUL_DESC_A_SCALE_MODE` (31) | A 的 scale 因子解释方式。 | `int32_t` | — |
| `ACLBLASLT_MATMUL_DESC_B_SCALE_MODE` (32) | B 的 scale 因子解释方式。 | `int32_t` | — |
| `ACLBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_A_EXT` (100) | 输入 A 参与计算的数据类型。 | `int32_t` | — |
| `ACLBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_B_EXT` (101) | 输入 B 参与计算的数据类型。 | `int32_t` | — |
| `ACLBLASLT_MATMUL_DESC_EPILOGUE_ACT_ARG0_EXT` (102) | 激活函数的第一个附加参数。 | `float` | — |
| `ACLBLASLT_MATMUL_DESC_EPILOGUE_ACT_ARG1_EXT` (103) | 激活函数的第二个附加参数。 | `float` | — |

### aclblasLtMatrixTransformDescAttribute_t

矩阵变换描述符的可配置属性，用于 `aclblasLtMatrixTransformDescSetAttribute` / `aclblasLtMatrixTransformDescGetAttribute`。

| 取值 | 含义 | 数据类型 | 默认值 |
|---|---|---|---|
| `ACLBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE` (0) | 计算（scale）数据类型，创建时指定。 | `int32_t`（`aclDataType`） | 创建时指定 |
| `ACLBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE` (1) | alpha / beta 的传递方式。 | `int32_t` | host |
| `ACLBLASLT_MATRIX_TRANSFORM_DESC_TRANSA` (2) | 对矩阵 A 的变换操作，见 `aclblasOperation_t`。 | `int32_t` | `ACLBLAS_OP_N` |
| `ACLBLASLT_MATRIX_TRANSFORM_DESC_TRANSB` (3) | 对矩阵 B 的变换操作，见 `aclblasOperation_t`。 | `int32_t` | `ACLBLAS_OP_N` |

### aclblasLtMatmulPreferenceAttribute_t

矩阵乘法算法搜索偏好的可配置属性，用于 `aclblasLtMatmulPreferenceSetAttribute` / `aclblasLtMatmulPreferenceGetAttribute`。

| 取值 | 含义 | 数据类型 | 默认值 |
|---|---|---|---|
| `ACLBLASLT_MATMUL_PREF_SEARCH_MODE` (0) | 搜索模式：0=启发式，1=穷举，2=快速。 | `uint32_t` | 0 |
| `ACLBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES` (1) | 允许的最大 workspace 字节数。 | `uint64_t` | 32MB（`32*1024*1024`） |

### aclblasLtMatmulTile_t

矩阵乘法 L1 Tile 的 M×N 形状枚举。仅包含内部 tile 候选列表中实际存在的 M×N 组合；K 维度不编码（受 `PackedAlgo` 16 字节布局限制）。用于 `ACLBLASLT_ALGO_CONFIG_TILE_ID` 属性。

| 取值 | 含义 |
|---|---|
| `ACLBLASLT_MATMUL_TILE_UNDEFINED` (0) | 未定义，由 heuristic 决定。 |
| `ACLBLASLT_MATMUL_TILE_128x128` (1) | 128×128（l1mDiv16=8, l1nDiv16=8）。 |
| `ACLBLASLT_MATMUL_TILE_128x256` (2) | 128×256（l1mDiv16=8, l1nDiv16=16）。 |
| `ACLBLASLT_MATMUL_TILE_256x128` (3) | 256×128（l1mDiv16=16, l1nDiv16=8）。 |
| `ACLBLASLT_MATMUL_TILE_256x256` (4) | 256×256（l1mDiv16=16, l1nDiv16=16）。 |

### aclblasLtMatmulStages_t

矩阵乘法流水级数（buffer 深度）枚举。映射到 `PackedAlgo.numBuffers`。用于 `ACLBLASLT_ALGO_CONFIG_STAGES_ID` 属性。

| 取值 | 含义 |
|---|---|
| `ACLBLASLT_MATMUL_STAGES_UNDEFINED` (0) | 自动选择（Set 时为 no-op）。 |
| `ACLBLASLT_MATMUL_STAGES_1` (1) | 单缓冲。 |
| `ACLBLASLT_MATMUL_STAGES_2` (2) | 双缓冲。 |
| `ACLBLASLT_MATMUL_STAGES_3` (3) | 三级流水。 |
| `ACLBLASLT_MATMUL_STAGES_4` (4) | 四级流水。 |

### aclblasLtReductionScheme_t

SplitK 归约方案枚举。映射到 `PackedAlgo.flags` 低 2 位。用于 `ACLBLASLT_ALGO_CONFIG_REDUCTION_SCHEME` 属性。Phase 1 仅完成 flags 字段读写，matmul 计算路径尚未消费。

| 取值 | 含义 |
|---|---|
| `ACLBLASLT_REDUCTION_SCHEME_NONE` (0) | 无归约（splitK=1）。 |
| `ACLBLASLT_REDUCTION_SCHEME_INPLACE` (1) | 原地归约（预留）。 |
| `ACLBLASLT_REDUCTION_SCHEME_WORKSPACE` (2) | 使用 workspace 归约（预留）。 |

### aclblasLtMatmulAlgoConfigAttributes_t

矩阵乘法算法配置属性枚举，用于 `aclblasLtMatmulAlgoConfigSetAttribute` / `aclblasLtMatmulAlgoConfigGetAttribute`。

| 取值 | 含义 | 数据类型 | 默认值 |
|---|---|---|---|
| `ACLBLASLT_ALGO_CONFIG_ID` (0) | 只读。算法索引，由 AlgoInit / ConfigSet 维护。 | `int32_t` | `0x00808201` |
| `ACLBLASLT_ALGO_CONFIG_TILE_ID` (1) | L1 Tile M×N 形状，见 `aclblasLtMatmulTile_t`。 | `uint32_t` | `TILE_128x128` |
| `ACLBLASLT_ALGO_CONFIG_STAGES_ID` (2) | 流水级数，见 `aclblasLtMatmulStages_t`。 | `uint32_t` | `STAGES_1` |
| `ACLBLASLT_ALGO_CONFIG_SPLITK_NUM` (3) | K 维切分数，范围 [1, 255]。 | `uint32_t` | 1 |
| `ACLBLASLT_ALGO_CONFIG_REDUCTION_SCHEME` (4) | SplitK 归约方案，见 `aclblasLtReductionScheme_t`。 | `uint32_t` | `NONE` |
| `ACLBLASLT_ALGO_CONFIG_CUSTOM_OPTION` (5) | dispatch policy（0=sync, 1=pingpong, 2=multi_stage）。 | `uint32_t` | 0 |
| `ACLBLASLT_ALGO_CONFIG_INNER_SHAPE_ID` (6) | 内部 MMA 形状（当前不支持，Set/Get 返回 `NOT_SUPPORTED`）。 | `uint16_t` | — |

## aclBLASLt Function Reference

aclBLASLt 提供描述符风格的矩阵乘法与矩阵变换能力，支持多种数据类型、内存布局（order）及 epilogue 融合，适用于大模型推理等场景。

| 接口名 | 说明 |
|---|---|
| [aclblasLtMatmul](../../blasLt/README.md) | 矩阵乘法 D = α·op(A)·op(B) + β·C，支持 FP32/MXFP8/MXFP4 等 |
| [aclblasLtMatrixTransform](../../blasLt/matrixtransform/README.md) | 矩阵转置、缩放、加法及 layout/dtype 转换 |

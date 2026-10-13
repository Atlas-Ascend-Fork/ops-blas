# aclblasCgetrfBatched（Ascend 950PR）

批量单精度复数 LU 分解。输入为列主序 complex64 方阵，原地写回 L/U；L 的单位对角线不单独存储。支持主元和非主元两种模式。

## 接口

公共声明位于 `include/cann_ops_blas.h`：

```cpp
aclblasStatus_t aclblasCgetrfBatched(
    aclblasHandle_t handle, int n, aclblasComplex* const Aarray[],
    int lda, int* PivotArray, int* infoArray, int batchSize);
```

| 参数 | 含义 |
| --- | --- |
| handle | 有效算子句柄，使用其中的 stream 异步执行 |
| n | 每个方阵的阶数，要求 n ≥ 0 |
| Aarray | Device 上的矩阵指针数组，各矩阵也位于 Device 且不能重叠 |
| lda | 列间跨度，要求 lda ≥ max(1, n)，支持列间 padding |
| PivotArray | Device 上 n × batchSize 个 1-based 主元编号；为空时不选主元 |
| infoArray | Device 上 batchSize 个整数；0 为成功，正值为首个零主元位置 |
| batchSize | 批次数量，要求 batchSize ≥ 0 |

主元按当前列 `|re| + |im|` 最大值选择，并列时取第一行。奇异矩阵仍继续处理，接口返回 SUCCESS，具体状态由 info 表达。主元模式要求 infoArray 非空；非主元模式允许 infoArray 为空。

先检查 handle、n、batchSize、lda，再处理 n=0 或 batchSize=0 的无操作返回。空 handle 返回 HANDLE_IS_NULLPTR，非法参数返回 INVALID_VALUE。读取结果前应同步 stream。

实现按向量宽度、片上容量和连续尺寸范围选择处理方式，不按测试编号、随机种子或指定阶数匹配。


## 支持的产品

| 产品 | 支持情况 |
| --- | --- |
| Ascend 950PR（arch35） | 支持 |
| Atlas A2/A3（arch22） | 本实现不支持 |
| Ascend 310P（arch20） | 本实现不支持 |

## 调用示例

下面的调用片段假定调用方已创建设备上下文、handle 和 stream，并准备好设备内存。
`deviceMatrices` 本身及其每个矩阵指针均指向设备内存；矩阵按列主序存储。
`devicePivots` 至少容纳 `n * batchSize` 个整数，`deviceInfo` 至少容纳 `batchSize` 个整数。

```cpp
#include "acl/acl.h"
#include "cann_ops_blas.h"

aclblasStatus_t FactorBatch(
    aclblasHandle_t handle, aclrtStream stream, int n,
    aclblasComplex* const deviceMatrices[], int lda,
    int* devicePivots, int* deviceInfo, int batchSize)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    const auto status = aclblasSetStream(handle, stream);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    return aclblasCgetrfBatched(
        handle, n, deviceMatrices, lda, devicePivots, deviceInfo, batchSize);
}
```

调用成功表示计算已提交；调用方应在读取结果或释放上述设备内存前，检查
`aclrtSynchronizeStream(stream)` 的返回值。随后将矩阵、主元及状态复制回主机，
检查每个 `deviceInfo` 对应的结果。传入空 `devicePivots` 可使用无主元模式。

矩阵地址和批量主元地址按 64 位计算。若列间 padding 超出 DMA 步长字段的表示范围，
自动采用支持该跨度的通用路径；输入矩阵和输出缓冲区仍须由调用方完整分配。

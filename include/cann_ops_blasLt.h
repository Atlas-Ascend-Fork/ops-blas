/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "cann_ops_blas_common.h"
#include <acl/acl.h>

#ifdef __cplusplus
#include <cstddef>
#include <cstdint>
#include <cstdio>
#else
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup types_module
 *  \brief Descriptor of the library context.
 */
typedef struct _aclblaslt_handle* aclblasLtHandle_t;

/*! \ingroup types_module
 *  \brief Descriptor of the matrix layout.
 */
typedef struct {
  uint64_t data[8];
} aclblasLtMatrixLayoutOpaque_t;

typedef aclblasLtMatrixLayoutOpaque_t* aclblasLtMatrixLayout_t;

/*! \ingroup types_module
 *  \brief Descriptor of the matmul operation.
 */
typedef struct {
  uint64_t data[24];
} aclblasLtMatmulDescOpaque_t;

typedef aclblasLtMatmulDescOpaque_t* aclblasLtMatmulDesc_t;

/*! \ingroup types_module
 *  \brief Descriptor of the matrix transform operation.
 */
typedef struct {
  uint64_t data[8];
} aclblasLtMatrixTransformDescOpaque_t;

typedef aclblasLtMatrixTransformDescOpaque_t* aclblasLtMatrixTransformDesc_t;

/*! \ingroup types_module
 *  \brief Descriptor of the matmul preference.
 */
typedef struct {
  uint64_t data[8];
} aclblasLtMatmulPreferenceOpaque_t;

typedef aclblasLtMatmulPreferenceOpaque_t* aclblasLtMatmulPreference_t;

/*! \ingroup types_module
 *  \struct aclblasLtMatmulAlgo_t
 *  \brief Description of the matrix multiplication algorithm.
 */
typedef struct _aclblasLtMatmulAlgo_t {
#ifdef __cplusplus
  uint8_t data[16] = {0};
  size_t max_workspace_bytes = 0;
#else
  uint8_t data[16];
  size_t max_workspace_bytes;
#endif
} aclblasLtMatmulAlgo_t;

/*! \ingroup types_module
 *  \brief Matmul Heuristic Result.
 */
typedef struct _aclblasLtMatmulHeuristicResult_t {
  aclblasLtMatmulAlgo_t algo; /**< Algo struct */
  size_t workspaceSize = 0;       /**< Actual size of workspace memory required. */
  aclblasStatus_t state = ACLBLAS_STATUS_SUCCESS;      /**< Result status. */
  float wavesCount = 1.0;           /**< Waves count is a device utilization metric. */
  int reserved[4];            /**< Reserved. */
} aclblasLtMatmulHeuristicResult_t;

/*! \ingroup types_module
 *  \brief Matrix order.
 */
typedef enum aclblasLtOrder {
  ACLBLASLT_ORDER_COL = 0, /**< Column major */
  ACLBLASLT_ORDER_ROW = 1, /**< Row major */
  ACLBLASLT_ORDER_COL32 = 2,        /**< 32-column composite tiles, column major within each tile. */
  ACLBLASLT_ORDER_COL4_4R2_8C = 3,  /**< 32-column 8-row composite quantization tiles. */
  ACLBLASLT_ORDER_COL32_2R_4R4 = 4, /**< 32-column 32-row composite quantization tiles. */
} aclblasLtOrder_t;

/*! \ingroup types_module
 *  \brief Storage location used for BLASLt alpha and beta scalar pointers.
 */
typedef enum aclblasLtPointerMode {
  ACLBLASLT_POINTER_MODE_HOST = 0,   /**< alpha and beta point to host scalars. */
  ACLBLASLT_POINTER_MODE_DEVICE = 1, /**< alpha and beta point to device scalars. */
} aclblasLtPointerMode_t;

/*! \ingroup types_module
 *  \brief  Property type.
 */
typedef enum aclblasLtPropertyType {
  ACLBLASLT_PROPERTY_MAJOR_VERSION = 0, /**<Major version number. */
  ACLBLASLT_PROPERTY_MINOR_VERSION = 1, /**<Minor version number. */
  ACLBLASLT_PROPERTY_PATCH_LEVEL = 2,   /**<Patch version number. */
} aclblasLtPropertyType_t;

/*! \ingroup types_module
 *  \brief Specifies the enumeration type to set the postprocessing options for the epilogue.
 */
typedef enum aclblasLtEpilogue {
  ACLBLASLT_EPILOGUE_DEFAULT = 1,                 /**<No special postprocessing. Scale and quantize the results if necessary.*/
  ACLBLASLT_EPILOGUE_RELU = 2,                    /**<Apply ReLU pointwise transform to the results (``x:=max(x, 0)``)*/
  ACLBLASLT_EPILOGUE_BIAS = 4,                    /**<Apply (broadcast) bias from the bias vector. The bias vector length must match the number of rows in matrix D, and it must be packed (so the stride between vector elements is one). The bias vector is broadcast to all columns and added before applying the final postprocessing.*/
  ACLBLASLT_EPILOGUE_RELU_BIAS = 6,               /**<Apply bias and then ReLU transform.*/
  ACLBLASLT_EPILOGUE_GELU = 32,                   /**<Apply GELU pointwise transform to the results (``x:=GELU(x)``).*/
  ACLBLASLT_EPILOGUE_GELU_BIAS = 36,              /**<Apply Bias and then GELU transform.*/
  ACLBLASLT_EPILOGUE_RELU_AUX = 130,              /**<Output GEMM results before applying RELU transform.*/
  ACLBLASLT_EPILOGUE_RELU_AUX_BIAS = 134,         /**<Output GEMM results after applying bias but before applying RELU transform.*/
  ACLBLASLT_EPILOGUE_DRELU = 136,
  ACLBLASLT_EPILOGUE_DRELU_BGRAD = 152,           /**<Apply gradient RELU transform and bias gradient to the results. Requires additional auxiliary input. */           /**<Apply gradient RELU transform. Requires additional auxiliary input. */
  ACLBLASLT_EPILOGUE_GELU_AUX = 160,              /**<Output GEMM results before applying GELU transform.*/
  ACLBLASLT_EPILOGUE_GELU_AUX_BIAS = 164,         /**<Output GEMM results after applying bias but before applying GELU transform.*/
  ACLBLASLT_EPILOGUE_DGELU = 192,                 /**<Apply gradient GELU transform. Requires additional auxiliary input. */
  ACLBLASLT_EPILOGUE_DGELU_BGRAD = 208,           /**<Apply gradient GELU transform and bias gradient to the results. Requires additional auxiliary input. */
  ACLBLASLT_EPILOGUE_BGRADA = 256,                /**<Apply bias gradient to A and output GEMM result. */
  ACLBLASLT_EPILOGUE_BGRADB = 512,                /**<Apply bias gradient to B and output GEMM result. */
  ACLBLASLT_EPILOGUE_SIGMOID = 1024,              /**<Apply sigmoid activation function pointwise. */
  ACLBLASLT_EPILOGUE_SWISH_EXT = 65536,           /**<Apply Swish pointwise transform to the results (``x:=Swish(x, 1)``).*/
  ACLBLASLT_EPILOGUE_SWISH_BIAS_EXT = 65540,      /**<Apply Bias and then Swish transform.*/
  ACLBLASLT_EPILOGUE_CLAMP_EXT = 131072,          /**<Apply pointwise clamp to the results (``x:=max(alpha, min(x, beta))``).*/
  ACLBLASLT_EPILOGUE_CLAMP_BIAS_EXT = 131076,     /**<Apply Bias and then clamp.*/
  ACLBLASLT_EPILOGUE_CLAMP_AUX_EXT = 131200,      /**<Output GEMM results before applying clamp transform.*/
  ACLBLASLT_EPILOGUE_CLAMP_AUX_BIAS_EXT = 131204, /**<Output GEMM results after applying bias but before applying clamp transform.*/
} aclblasLtEpilogue_t;

/*! \ingroup types_module
 *  \brief Matrix layout attributes.
 */
typedef enum aclblasLtMatrixLayoutAttribute {
  ACLBLASLT_MATRIX_LAYOUT_BATCH_COUNT = 0,          /**< Batch count. Default: 1. Type: ``int32_t``. */
  ACLBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET = 1, /**< Strided-batch offset (elements). Default: 0. Type: ``int64_t``. */
  ACLBLASLT_MATRIX_LAYOUT_TYPE = 2,                 /**< Matrix data type. See ``aclDataType``. Type: ``uint32_t``. */
  ACLBLASLT_MATRIX_LAYOUT_ORDER = 3,                /**< Memory order. See ``aclblasLtOrder_t``. Default: ``ACLBLASLT_ORDER_COL``. Type: ``int32_t``. */
  ACLBLASLT_MATRIX_LAYOUT_ROWS = 4,                 /**< Row count. Type: ``uint64_t``. */
  ACLBLASLT_MATRIX_LAYOUT_COLS = 5,                 /**< Column count. Type: ``uint64_t``. */
  ACLBLASLT_MATRIX_LAYOUT_LD = 6,                   /**< Leading dimension (elements). Type: ``int64_t``. */
} aclblasLtMatrixLayoutAttribute_t;

/*! \ingroup types_module
 *  \brief Matmul operation descriptor attributes.
 */
typedef enum aclblasLtMatmulDescAttribute {
  ACLBLASLT_MATMUL_DESC_TRANSA = 0,                     /**<Specifies the type of transformation operation that should be performed on matrix A. Default value is ``ACLBLAS_OP_N`` (for example, non-transpose operation). See ``aclblasOperation_t``. Data type: ``int32_t``. */
  ACLBLASLT_MATMUL_DESC_TRANSB = 1,                     /**<Specifies the type of transformation operation that should be performed on matrix B. Default value is ``ACLBLAS_OP_N`` (for example, non-transpose operation). See ``aclblasOperation_t``. Data type: ``int32_t``. */
  ACLBLASLT_MATMUL_DESC_EPILOGUE = 2,                   /**<Epilogue function. See ``aclblasLtEpilogue_t``. Default value is ``ACLBLASLT_EPILOGUE_DEFAULT``. Data type: ``uint32_t``. */
  ACLBLASLT_MATMUL_DESC_BIAS_POINTER = 3,               /**<Bias or bias gradient vector pointer in the device memory. Data type: ``void*`` / ``const void*``. */
  ACLBLASLT_MATMUL_DESC_BIAS_DATA_TYPE = 4,             /**<Type of the bias vector in the device memory. Can be set the same as the D matrix type or Scale type. Bias case: see ``ACLBLASLT_EPILOGUE_BIAS``. Data type: ``int32_t`` based on ``aclDataType``. */
  ACLBLASLT_MATMUL_DESC_A_SCALE_POINTER = 5,            /**<Device pointer to the scale factor value that converts data in matrix A to the compute data type range. The scaling factor must have the same type as the compute type. If not specified, or set to NULL, the scaling factor is assumed to be ``1``. If set for an unsupported matrix data, scale, and compute type combination, calling aclblasLtMatmul() will return ``ACLBLAS_INVALID_VALUE``. Default value: NULL. Data type: ``void*`` ``/const void*``. */
  ACLBLASLT_MATMUL_DESC_B_SCALE_POINTER = 6,            /**<Equivalent to ``ACLBLASLT_MATMUL_DESC_A_SCALE_POINTER`` for matrix B. Default value: NULL. Data type: ``void*`` / ``const void*``. */
  ACLBLASLT_MATMUL_DESC_C_SCALE_POINTER = 7,            /**<Equivalent to ``ACLBLASLT_MATMUL_DESC_A_SCALE_POINTER`` for matrix C. Default value: NULL. Data type: ``void*`` / ``const void*``. */
  ACLBLASLT_MATMUL_DESC_D_SCALE_POINTER = 8,            /**<Equivalent to ``ACLBLASLT_MATMUL_DESC_A_SCALE_POINTER`` for matrix D. Default value: NULL. Data type: ``void*`` / ``const void*``. */
  ACLBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER = 9, /**<Equivalent to ``ACLBLASLT_MATMUL_DESC_A_SCALE_POINTER`` for matrix AUX. Default value: NULL. Data type: ``void*`` / ``const void*``. */
  ACLBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER = 10,      /**<Epilogue auxiliary buffer pointer in the device memory. Data type: ``void*`` / ``const void*``. */
  ACLBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD = 11,           /**<The leading dimension of the epilogue auxiliary buffer pointer in the device memory. Data type: ``int64_t``. */
  ACLBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE = 12, /**<The batch stride of the epilogue auxiliary buffer pointer in the device memory. Data type: ``int64_t``. */
  ACLBLASLT_MATMUL_DESC_POINTER_MODE = 13,              /**<Specifies whether alpha and beta point to host or device scalars. Default: ``ACLBLASLT_POINTER_MODE_HOST``. Data type: ``aclblasLtPointerMode_t``. */
  ACLBLASLT_MATMUL_DESC_AMAX_D_POINTER = 14,           /**<Device pointer to the memory that on completion will be set to the maximum of the absolute values in the output matrix. Data type: ``void*`` / ``const void*``. */
  ACLBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE = 22,    /**<Type of the auxiliary vector in the device memory. Default value is: ``ACLBLASLT_DATATYPE_INVALID`` (using D matrix type). Data type: ``int32_t`` based on ``aclDataType``. */
  ACLBLASLT_MATMUL_DESC_A_SCALE_MODE = 31,                   /**<Scaling mode that defines how the matrix scaling factor for matrix A is interpreted. See ``aclblasLtMatmulMatrixScale_t``. */
  ACLBLASLT_MATMUL_DESC_B_SCALE_MODE = 32,                   /**<Scaling mode that defines how the matrix scaling factor for matrix B is interpreted. See ``aclblasLtMatmulMatrixScale_t``. */
  ACLBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_A_EXT = 100,     /**<Compute input A types. Defines the data type used for the input A of a matrix multiply. */
  ACLBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_B_EXT = 101,           /**<Compute input B types. Defines the data type used for the input B of a matrix multiply. */
  ACLBLASLT_MATMUL_DESC_EPILOGUE_ACT_ARG0_EXT = 102,              /**<First extra argument for the activation function. Data type: ``float``. */
  ACLBLASLT_MATMUL_DESC_EPILOGUE_ACT_ARG1_EXT = 103,              /**<Second extra argument for the activation function. Data type: ``float``. */
  ACLBLASLT_MATMUL_DESC_MAX,
} aclblasLtMatmulDescAttribute_t;

/*! \ingroup types_module
 *  \brief Matrix transform descriptor attributes.
 */
typedef enum aclblasLtMatrixTransformDescAttribute {
  ACLBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE = 0,   /**<Scale (compute) data type. Set at create time. Data type: ``int32_t`` based on ``aclDataType``. */
  ACLBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE = 1, /**<Pointer mode for alpha / beta. Default host. Data type: ``int32_t``. */
  ACLBLASLT_MATRIX_TRANSFORM_DESC_TRANSA = 2,       /**<Transform operation on matrix A. Default ``ACLBLAS_OP_N``. See ``aclblasOperation_t``. Data type: ``int32_t``. */
  ACLBLASLT_MATRIX_TRANSFORM_DESC_TRANSB = 3,       /**<Transform operation on matrix B. Default ``ACLBLAS_OP_N``. See ``aclblasOperation_t``. Data type: ``int32_t``. */
} aclblasLtMatrixTransformDescAttribute_t;

/*! \ingroup types_module
 *  \brief Matmul preference attributes.
 */
typedef enum aclblasLtMatmulPreferenceAttribute {
  ACLBLASLT_MATMUL_PREF_SEARCH_MODE = 0,          /**<Search mode. Data type: ``uint32_t``. */
  ACLBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES = 1,  /**<Maximum allowed workspace memory. Default is 32MB (32*1024*1024 bytes). Data type: ``uint64_t``. */
  ACLBLASLT_MATMUL_PREF_MAX = 2
} aclblasLtMatmulPreferenceAttribute_t;

/*! \ingroup types_module
 *  \brief Matmul algorithm L1 tile shapes (M x N).
 *
 *  Only M x N combinations present in the internal tile candidate list are
 *  exported. K dimension is not encoded (see PackedAlgo layout limitation).
 */
typedef enum aclblasLtMatmulTile {
    ACLBLASLT_MATMUL_TILE_UNDEFINED = 0, /**< Undefined / determined by heuristic. */
    ACLBLASLT_MATMUL_TILE_128x128 = 1,   /**< 128 x 128 (l1mDiv16=8,  l1nDiv16=8).  */
    ACLBLASLT_MATMUL_TILE_128x256 = 2,   /**< 128 x 256 (l1mDiv16=8,  l1nDiv16=16). */
    ACLBLASLT_MATMUL_TILE_256x128 = 3,   /**< 256 x 128 (l1mDiv16=16, l1nDiv16=8).  */
    ACLBLASLT_MATMUL_TILE_256x256 = 4,   /**< 256 x 256 (l1mDiv16=16, l1nDiv16=16). */
} aclblasLtMatmulTile_t;

/*! \ingroup types_module
 *  \brief Matmul pipeline stages (buffer depth).
 */
typedef enum aclblasLtMatmulStages {
    ACLBLASLT_MATMUL_STAGES_UNDEFINED = 0, /**< Auto-select. */
    ACLBLASLT_MATMUL_STAGES_1 = 1,          /**< Single buffer. */
    ACLBLASLT_MATMUL_STAGES_2 = 2,          /**< Double buffer. */
    ACLBLASLT_MATMUL_STAGES_3 = 3,          /**< Three-stage pipeline. */
    ACLBLASLT_MATMUL_STAGES_4 = 4,          /**< Four-stage pipeline. */
} aclblasLtMatmulStages_t;

/*! \ingroup types_module
 *  \brief SplitK reduction schemes.
 */
typedef enum aclblasLtReductionScheme {
    ACLBLASLT_REDUCTION_SCHEME_NONE = 0,      /**< No reduction (splitK=1). */
    ACLBLASLT_REDUCTION_SCHEME_INPLACE = 1,    /**< In-place reduction (reserved). */
    ACLBLASLT_REDUCTION_SCHEME_WORKSPACE = 2,  /**< Workspace-based reduction (reserved). */
} aclblasLtReductionScheme_t;

/*! \ingroup types_module
 *  \brief Matmul algorithm configuration attributes.
 *
 *  Maps to fields of the internal PackedAlgo structure. Attributes marked
 *  read-only or unsupported return ACLBLAS_STATUS_NOT_SUPPORTED from
 *  SetAttribute / GetAttribute.
 */
typedef enum aclblasLtMatmulAlgoConfigAttributes {
    ACLBLASLT_ALGO_CONFIG_ID = 0,            /**< Read-only. Algorithm index. Type: int32_t. */
    ACLBLASLT_ALGO_CONFIG_TILE_ID = 1,       /**< L1 Tile M x N shape. Default UNDEFINED. Type: uint32_t. */
    ACLBLASLT_ALGO_CONFIG_STAGES_ID = 2,     /**< Pipeline stages. Default UNDEFINED. Type: uint32_t. */
    ACLBLASLT_ALGO_CONFIG_SPLITK_NUM = 3,    /**< K split count. Default 1. Type: uint32_t. */
    ACLBLASLT_ALGO_CONFIG_REDUCTION_SCHEME = 4, /**< SplitK reduction scheme. Default NONE. Type: uint32_t. */
    ACLBLASLT_ALGO_CONFIG_CUSTOM_OPTION = 5, /**< Dispatch policy. Type: uint32_t. */
    ACLBLASLT_ALGO_CONFIG_INNER_SHAPE_ID = 6, /**< Inner MMA shape (not supported). Type: uint16_t. */
    ACLBLASLT_ALGO_CONFIG_MAX,
} aclblasLtMatmulAlgoConfigAttributes_t;

/*! \ingroup type_module
 *  \brief Logger callback function pointer type.
 *
 *  When set via aclblasLtLoggerSetCallback(), the library invokes this
 *  function for each log message instead of writing to a file or stdout.
 *
 *  \param logLevel     The log level of this message. See aclblasLtLogLevel_t.
 *  \param functionName Name of the API function that produced this message.
 *  \param message      The log message text (null-terminated).
 */
typedef void (*aclblasLtLoggerCallback_t)(int logLevel, const char* functionName, const char* message);

/*! \ingroup type_module
 *  \brief Log level for aclblasLt logging.
 *
 *  Maps 1:1 to cuBLASLt's CUBLASLT_LOG_LEVEL values.
 */
typedef enum aclblasLtLogLevel {
    ACLBLASLT_LOG_LEVEL_OFF       = 0, /**< Logging disabled (default). */
    ACLBLASLT_LOG_LEVEL_ERROR     = 1, /**< Only errors. */
    ACLBLASLT_LOG_LEVEL_TRACE     = 2, /**< API calls that launch kernels. */
    ACLBLASLT_LOG_LEVEL_HINTS     = 3, /**< Performance hints. */
    ACLBLASLT_LOG_LEVEL_INFO      = 4, /**< General library execution info. */
    ACLBLASLT_LOG_LEVEL_API_TRACE = 5, /**< All API calls. */
} aclblasLtLogLevel_t;

/*! \ingroup type_module
 *  \brief Log mask bit flags for aclblasLt logging.
 *
 *  Combine with bitwise OR. Maps 1:1 to cuBLASLt's CUBLASLT_LOG_MASK values.
 */
typedef enum aclblasLtLogMask {
    ACLBLASLT_LOG_MASK_OFF       = 0,  /**< Off. */
    ACLBLASLT_LOG_MASK_ERROR     = 1,  /**< Error messages. */
    ACLBLASLT_LOG_MASK_TRACE     = 2,  /**< Kernel-launch trace. */
    ACLBLASLT_LOG_MASK_HINTS     = 4,  /**< Performance hints. */
    ACLBLASLT_LOG_MASK_INFO      = 8,  /**< General info. */
    ACLBLASLT_LOG_MASK_API_TRACE = 16, /**< API trace. */
} aclblasLtLogMask_t;

/*! \ingroup library_module
 *  \brief Query aclBLASLt packed version number.
 */
 aclblasStatus_t aclblasLtGetVersion(size_t* version);

 /*! \ingroup library_module
  *  \brief Query aclBLASLt property value.
  */
 aclblasStatus_t aclblasLtGetProperty(aclblasLtPropertyType_t type, int* value);

// Library management
/*! \ingroup library_module
 *  \brief Create a aclBLASLt handle.
 *
 *  \details
 *  This function initializes the aclBLASLt library and creates a handle to an
 *  opaque structure holding the aclBLASLt library context. It allocates light
 *  hardware resources on the host and device and must be called prior to making
 *  any other aclBLASLt library calls. The aclBLASLt library context is tied to
 *  the current CANN device. To use the library on multiple devices, one
 *  aclBLASLt handle should be created for each device.
 *
 *  @param[out]
 *  lightHandle Pointer to the allocated aclBLASLt handle for the created aclBLASLt
 *  context.
 *
 *  \retval ACLBLAS_STATUS_SUCCESS The allocation completed successfully.
 *  \retval ACLBLAS_STATUS_INVALID_VALUE \p lightHandle == NULL.
 *  \retval ACLBLAS_STATUS_NOT_INITIALIZED The CANN runtime is not initialized.
 *  \retval ACLBLAS_STATUS_ALLOC_FAILED If memory allocation fails.
 */
aclblasStatus_t aclblasLtCreate(aclblasLtHandle_t* lightHandle);

/*! \ingroup library_module
 *  \brief Destroy a aclBLASLt handle.
 *
 *  \details
 *  This function releases hardware resources used by the aclBLASLt library.
 *  It is usually the last call with a particular handle to the
 *  aclBLASLt library. Because aclblasLtCreate() allocates some internal
 *  resources and the release of those resources by calling aclblasLtDestroy()
 *  implicitly calls device synchronization, it is recommended to minimize
 *  the number of aclblasLtCreate() / aclblasLtDestroy() occurrences.
 *
 *  @param[in]
 *  lightHandle Pointer to the aclBLASLt handle to be destroyed.
 *
 *  \retval ACLBLAS_STATUS_SUCCESS The aclBLASLt context was successfully
 *  destroyed.
 *  \retval ACLBLAS_STATUS_INVALID_VALUE \p lightHandle == NULL or is not a valid aclBLASLt handle.
 */
aclblasStatus_t aclblasLtDestroy(const aclblasLtHandle_t lightHandle);

// Matrix layout descriptor
/*! \ingroup library_module
 *  \brief Create a matrix layout descriptor.
 *
 *  \details
 *  This function creates a matrix layout descriptor by allocating the memory
 *  needed to hold its opaque structure.
 *
 *  @param[out]
 *  matLayout Pointer to the structure holding the matrix layout descriptor
 *  created by this function. See \ref aclblasLtMatrixLayout_t.
 *  @param[in]
 *  type Enumerant that specifies the data precision for the matrix layout
 *  descriptor created by this function. See aclDataType.
 *  @param[in]
 *  rows Number of rows of the matrix.
 *  @param[in]
 *  cols Number of columns of the matrix.
 *  @param[in]
 *  ld The leading dimension of the matrix. In column major layout, this is the
 *  number of elements to jump to reach the next column. Therefore, ld >= m (number of
 *  rows).
 *
 *  \retval ACLBLAS_STATUS_SUCCESS If the descriptor was created successfully.
 *  \retval ACLBLAS_STATUS_INVALID_VALUE If \p matLayout is NULL or \p ld is negative.
 *  \retval ACLBLAS_STATUS_ALLOC_FAILED If the memory could not be allocated.
 */
aclblasStatus_t aclblasLtMatrixLayoutCreate(aclblasLtMatrixLayout_t* matLayout,
                                            aclDataType type,
                                            uint64_t rows,
                                            uint64_t cols,
                                            int64_t ld);

/*! \ingroup library_module
 *  \brief Destroy a matrix layout descriptor.
 *
 *  \details
 *  This function destroys a previously created matrix layout descriptor object.
 *
 *  @param[in]
 *  matLayout Pointer to the structure holding the matrix layout descriptor to
 *  be destroyed by this function. See \ref aclblasLtMatrixLayout_t.
 *
 *  \retval ACLBLAS_STATUS_SUCCESS If the operation was successful.
 *  \retval ACLBLAS_STATUS_INVALID_VALUE If \p matLayout is NULL.
 */
aclblasStatus_t aclblasLtMatrixLayoutDestroy(const aclblasLtMatrixLayout_t matLayout);

/*! \ingroup library_module
 *  \brief Set an attribute for a matrix descriptor.
 *
 *  \details
 *  This function sets the value of the specified attribute belonging to a
 *  previously created matrix descriptor.
 *
 *  @param[in]
 *  matLayout Pointer to the previously created structure holding the matrix
 *  descriptor queried by this function. See \ref aclblasLtMatrixLayout_t.
 *  @param[in]
 *  attr The attribute that will be set by this function. See \ref
 *  aclblasLtMatrixLayoutAttribute_t.
 *  @param[in]
 *  buf The value to which the specified attribute should be set.
 *  @param[in]
 *  sizeInBytes Size of the buf buffer (in bytes) for verification.
 *
 *  \retval ACLBLAS_STATUS_SUCCESS If the attribute was set successfully.
 *  \retval ACLBLAS_STATUS_INVALID_VALUE If \p buf is NULL or \p sizeInBytes
 *  doesn't match the size of the internal storage for the selected attribute.
 */
aclblasStatus_t aclblasLtMatrixLayoutSetAttribute(aclblasLtMatrixLayout_t matLayout,
                                                  aclblasLtMatrixLayoutAttribute_t attr,
                                                  const void* buf,
                                                  size_t sizeInBytes);

/*! \ingroup library_module
 *  \brief Get an attribute for a matrix descriptor.
 */
aclblasStatus_t aclblasLtMatrixLayoutGetAttribute(const aclblasLtMatrixLayout_t matLayout,
                                                  aclblasLtMatrixLayoutAttribute_t attr,
                                                  void* buf,
                                                  size_t sizeInBytes,
                                                  size_t* sizeWritten);

// Matmul operation descriptor
/*! \ingroup library_module
 *  \brief Create a matrix multiply descriptor.
 *
 *  \details
 *  This function creates a matrix multiply descriptor by allocating the memory
 *  needed to hold its opaque structure.
 *
 *  @param[out]
 *  matmulDesc Pointer to the structure holding the matrix multiply descriptor
 *  created by this function. See \ref aclblasLtMatmulDesc_t.
 *  @param[in]
 *  computeType Enumerant that specifies the data precision for the matrix
 *  multiply descriptor this function creates. See aclblasComputeType_t.
 *  @param[in]
 *  scaleType Enumerant that specifies the data precision for the matrix
 *  transform descriptor this function creates. See aclDataType.
 *
 *  \retval ACLBLAS_STATUS_SUCCESS If the descriptor was created successfully.
 *  \retval ACLBLAS_STATUS_INVALID_VALUE If \p matmulDesc is NULL.
 *  \retval ACLBLAS_STATUS_ALLOC_FAILED If the memory could not be allocated.
 */
aclblasStatus_t aclblasLtMatmulDescCreate(aclblasLtMatmulDesc_t* matmulDesc,
                                          aclblasComputeType_t computeType,
                                          aclDataType scaleType);

/*! \ingroup library_module
 *  \brief Destroy a matrix multiply descriptor.
 *
 *  \details
 *  This function destroys a previously created matrix multiply descriptor
 *  object.
 *
 *  @param[in]
 *  matmulDesc Pointer to the structure holding the matrix multiply descriptor
 *  to be destroyed by this function. See \ref aclblasLtMatmulDesc_t.
 *
 *  \retval ACLBLAS_STATUS_SUCCESS If operation was successful.
 *  \retval ACLBLAS_STATUS_INVALID_VALUE If \p matmulDesc is NULL.
 */
aclblasStatus_t aclblasLtMatmulDescDestroy(const aclblasLtMatmulDesc_t matmulDesc);

/*! \ingroup library_module
 *  \brief Initialize a caller-allocated matmul descriptor capsule.
 *
 *  \details
 *  This function initializes a matrix multiply descriptor that the caller has
 *  already allocated (e.g. on the stack). Unlike aclblasLtMatmulDescCreate(),
 *  this function does not allocate heap memory and the resulting capsule must
 *  NOT be passed to aclblasLtMatmulDescDestroy().
 *
 *  @param[out]
 *  matmulDesc Pointer to the caller-allocated capsule to initialize.
 *  @param[in]
 *  computeType Enumerant that specifies the compute precision.
 *  @param[in]
 *  scaleType Enumerant that specifies the scale data type.
 *
 *  \retval ACLBLAS_STATUS_SUCCESS If initialization was successful.
 *  \retval ACLBLAS_STATUS_INVALID_VALUE If \p matmulDesc is NULL.
 *  \retval ACLBLAS_STATUS_INTERNAL_ERROR If an internal copy fails.
 */
aclblasStatus_t aclblasLtMatmulDescInit(aclblasLtMatmulDesc_t matmulDesc,
                                        aclblasComputeType_t computeType,
                                        aclDataType scaleType);

/*! \ingroup library_module
 *  \brief Set attribute to a matrix multiply descriptor.
 *
 *  \details
 *  This function sets the value of the specified attribute belonging to a
 *  previously created matrix multiply descriptor.
 *
 *  @param[in]
 *  matmulDesc Pointer to the previously created structure holding the matrix
 *  multiply descriptor queried by this function. See \ref aclblasLtMatmulDesc_t.
 *  @param[in]
 *  attr The attribute that will be set by this function. See \ref
 *  aclblasLtMatmulDescAttribute_t.
 *  @param[in]
 *  buf The value to which the specified attribute should be set.
 *  @param[in]
 *  sizeInBytes Size of the buf buffer (in bytes) for verification.
 *
 *  \retval ACLBLAS_STATUS_SUCCESS If the attribute was set successfully.
 *  \retval ACLBLAS_STATUS_INVALID_VALUE If \p buf is NULL or \p sizeInBytes
 *  doesn't match the size of the internal storage for the selected attribute.
 *  \retval ACLBLAS_STATUS_NOT_SUPPORTED If \p attr is not a recognized attribute.
 */
aclblasStatus_t aclblasLtMatmulDescSetAttribute(aclblasLtMatmulDesc_t matmulDesc,
                                                aclblasLtMatmulDescAttribute_t attr,
                                                const void* buf,
                                                size_t sizeInBytes);

/*! \ingroup library_module
 *  \brief Get an attribute from a matrix multiply descriptor.
 *
 *  \details
 *  This function retrieves the value of the specified attribute from a
 *  previously created matrix multiply descriptor.
 *
 *  @param[in]
 *  desc Pointer to the previously created matrix multiply descriptor. See
 *  \ref aclblasLtMatmulDesc_t.
 *  @param[in]
 *  attr The attribute to query. See \ref aclblasLtMatmulDescAttribute_t.
 *  @param[out]
 *  buf Output buffer used to store the queried attribute value.
 *  @param[in]
 *  sizeInBytes Size of \p buf in bytes.
 *  @param[out]
 *  sizeWritten Number of bytes actually written to \p buf. Can be NULL.
 *
 *  \retval ACLBLAS_STATUS_SUCCESS If the attribute was retrieved successfully.
 *  \retval ACLBLAS_STATUS_INVALID_VALUE If \p desc or \p buf is NULL, or
 *  \p sizeInBytes is smaller than the required size for the selected
 *  attribute.
 *  \retval ACLBLAS_STATUS_NOT_SUPPORTED If \p attr is not a recognized attribute.
 */
aclblasStatus_t aclblasLtMatmulDescGetAttribute(aclblasLtMatmulDesc_t desc,
                                                aclblasLtMatmulDescAttribute_t attr,
                                                void* buf,
                                                size_t sizeInBytes,
                                                size_t* sizeWritten);

// Preference
/*! \ingroup library_module
 *  \brief Create a preference descriptor.
 *
 *  \details
 *  This function creates a matrix multiply heuristic search preferences
 *  descriptor by allocating the memory needed to hold its opaque structure.
 *
 *  @param[out]
 *  pref Pointer to the structure holding the matrix multiply preferences
 *  descriptor created by this function. see \ref aclblasLtMatmulPreference_t.
 *
 *  \retval ACLBLAS_STATUS_SUCCESS If the descriptor was created
 *  successfully.
 *  \retval ACLBLAS_STATUS_INVALID_VALUE If \p pref is NULL.
 *  \retval ACLBLAS_STATUS_ALLOC_FAILED If memory could not be
 *  allocated.
 */
aclblasStatus_t aclblasLtMatmulPreferenceCreate(aclblasLtMatmulPreference_t* pref);

/*! \ingroup library_module
 *  \brief Destroy a preference descriptor.
 *
 *  \details
 *  This function destroys a previously created matrix multiply preferences
 *  descriptor object.
 *
 *  @param[in]
 *  pref Pointer to the structure holding the matrix multiply preferences
 *  descriptor to be destroyed by this function. See \ref
 *  aclblasLtMatmulPreference_t.
 *
 *  \retval ACLBLAS_STATUS_SUCCESS If operation was successful.
 *  \retval ACLBLAS_STATUS_INVALID_VALUE If \p pref is NULL.
 */
aclblasStatus_t aclblasLtMatmulPreferenceDestroy(const aclblasLtMatmulPreference_t pref);

/*! \ingroup library_module
 *  \brief Set attribute in a preference descriptor.
 *
 *  \details
 *  This function sets the value of the specified attribute belonging to a
 *  previously created matrix multiply preferences descriptor.
 *
 *  @param[in]
 *  pref Pointer to the previously created structure holding the matrix
 *  multiply preferences descriptor queried by this function. See \ref
 *  aclblasLtMatmulPreference_t.
 *  @param[in]
 *  attr The attribute that will be set by this function. See \ref
 *  aclblasLtMatmulPreferenceAttribute_t.
 *  @param[in]
 *  buf The value to which the specified attribute should be set.
 *  @param[in]
 *  sizeInBytes Size of the \p buf buffer (in bytes) for verification.
 *
 *  \retval ACLBLAS_STATUS_SUCCESS If the attribute was set successfully.
 *  \retval ACLBLAS_STATUS_INVALID_VALUE If \p buf is NULL or \p sizeInBytes
 *  doesn't match the size of the internal storage for the selected attribute.
 *  \retval ACLBLAS_STATUS_NOT_SUPPORTED If \p attr is not a recognized attribute.
 */
aclblasStatus_t aclblasLtMatmulPreferenceSetAttribute(aclblasLtMatmulPreference_t pref,
                                                      aclblasLtMatmulPreferenceAttribute_t attr,
                                                      const void* buf,
                                                      size_t sizeInBytes);


/*! \ingroup library_module
 *  \brief Get an attribute from a preference descriptor.
 *
 *  \details
 *  This function retrieves the value of the specified attribute from a
 *  previously created matrix multiply preference descriptor.
 *
 *  @param[in]
 *  pref Pointer to the previously created preference descriptor. See
 *  \ref aclblasLtMatmulPreference_t.
 *  @param[in]
 *  attr The attribute to query. See
 *  \ref aclblasLtMatmulPreferenceAttribute_t.
 *  @param[out]
 *  buf Output buffer used to store the queried attribute value.
 *  @param[in]
 *  sizeInBytes Size of \p buf in bytes.
 *  @param[out]
 *  sizeWritten Number of bytes actually written to \p buf. Can be NULL.
 *
 *  \retval ACLBLAS_STATUS_SUCCESS If the attribute was retrieved successfully.
 *  \retval ACLBLAS_STATUS_INVALID_VALUE If \p pref or \p buf is NULL, or
 *  \p sizeInBytes is smaller than the required size for the selected
 *  attribute.
 *  \retval ACLBLAS_STATUS_NOT_SUPPORTED If \p attr is not a recognized attribute.
 */
aclblasStatus_t aclblasLtMatmulPreferenceGetAttribute(aclblasLtMatmulPreference_t pref,
                                                      aclblasLtMatmulPreferenceAttribute_t attr,
                                                      void* buf,
                                                      size_t sizeInBytes,
                                                      size_t* sizeWritten);

// Heuristic
/*! \ingroup library_module
 *  \brief Initialize a matmul algorithm structure for a given algorithm ID.
 *
 *  \details
 *  This function initializes an aclblasLtMatmulAlgo_t structure according to
 *  the supplied algorithm ID and data type combination. An algoId of 0 requests
 *  the default configuration.
 *
 *  @param[in]
 *  lightHandle Pointer to the allocated aclBLASLt handle.
 *  @param[in]
 *  computeType Compute precision. See aclblasComputeType_t.
 *  @param[in]
 *  scaleType Scale data type. See aclDataType.
 *  @param[in]
 *  Atype,Btype,Ctype,Dtype Data types of matrices A, B, C, D.
 *  @param[in]
 *  algoId Algorithm ID. 0 selects the default; non-zero must be a valid
 *  encoded ID.
 *  @param[out]
 *  algo Pointer to the algorithm structure to initialize.
 *
 *  \retval ACLBLAS_STATUS_SUCCESS If initialization was successful.
 *  \retval ACLBLAS_STATUS_NOT_INITIALIZED If \p lightHandle is NULL.
 *  \retval ACLBLAS_STATUS_INVALID_VALUE If \p algo is NULL, \p algoId is
 *  negative, or a non-zero algoId cannot be decoded.
 *  \retval ACLBLAS_STATUS_NOT_SUPPORTED If the data type combination is
 *  unsupported.
 */
aclblasStatus_t aclblasLtMatmulAlgoInit(aclblasLtHandle_t lightHandle,
                                        aclblasComputeType_t computeType,
                                        aclDataType scaleType,
                                        aclDataType Atype,
                                        aclDataType Btype,
                                        aclDataType Ctype,
                                        aclDataType Dtype,
                                        int algoId,
                                        aclblasLtMatmulAlgo_t* algo);

/*! \ingroup library_module
 *  \brief Set a configuration attribute on a matmul algorithm.
 *
 *  \details
 *  This function sets the value of the specified configuration attribute
 *  belonging to a previously initialized aclblasLtMatmulAlgo_t structure.
 *
 *  @param[in]
 *  algo Pointer to the algorithm structure to modify.
 *  @param[in]
 *  attr The attribute to set. See aclblasLtMatmulAlgoConfigAttributes_t.
 *  @param[in]
 *  buf Source buffer holding the attribute value.
 *  @param[in]
 *  sizeInBytes Size of \p buf in bytes.
 *
 *  \retval ACLBLAS_STATUS_SUCCESS If the attribute was set successfully.
 *  \retval ACLBLAS_STATUS_INVALID_VALUE If a pointer is NULL, size mismatches,
 *  value is out of range, or the algo magic is invalid.
 *  \retval ACLBLAS_STATUS_NOT_SUPPORTED If the attribute is read-only or
 *  unsupported.
 */
aclblasStatus_t aclblasLtMatmulAlgoConfigSetAttribute(aclblasLtMatmulAlgo_t* algo,
                                                      aclblasLtMatmulAlgoConfigAttributes_t attr,
                                                      const void* buf,
                                                      size_t sizeInBytes);

/*! \ingroup library_module
 *  \brief Query a configuration attribute from a matmul algorithm.
 *
 *  \details
 *  This function retrieves the value of the specified configuration attribute
 *  from an aclblasLtMatmulAlgo_t structure. When \p sizeInBytes is 0, the
 *  required buffer size is returned via \p sizeWritten and \p buf is not
 *  written.
 *
 *  @param[in]
 *  algo Pointer to the algorithm structure to query.
 *  @param[in]
 *  attr The attribute to query. See aclblasLtMatmulAlgoConfigAttributes_t.
 *  @param[out]
 *  buf Output buffer for the attribute value. May be NULL when sizeInBytes is 0.
 *  @param[in]
 *  sizeInBytes Size of \p buf in bytes.
 *  @param[out]
 *  sizeWritten Number of bytes written (or required). May be NULL.
 *
 *  \retval ACLBLAS_STATUS_SUCCESS If the attribute was retrieved successfully.
 *  \retval ACLBLAS_STATUS_INVALID_VALUE If a pointer is NULL, buffer is too
 *  small, or the algo magic is invalid.
 *  \retval ACLBLAS_STATUS_NOT_SUPPORTED If the attribute is unsupported.
 */
aclblasStatus_t aclblasLtMatmulAlgoConfigGetAttribute(const aclblasLtMatmulAlgo_t* algo,
                                                      aclblasLtMatmulAlgoConfigAttributes_t attr,
                                                      void* buf,
                                                      size_t sizeInBytes,
                                                      size_t* sizeWritten);

/*! \ingroup library_module
 *  \brief Retrieve the supported algorithm IDs for a matmul configuration.
 *
 *  \details
 *  This function returns the set of algorithm IDs supported by aclBLASLt for
 *  the given compute and data type combination. Each returned ID is a valid
 *  input to aclblasLtMatmulAlgoInit(). The IDs are returned in ascending
 *  order. If \p algoIdsArrayLength is smaller than the number of available
 *  IDs, only the first \p algoIdsArrayLength IDs are written, and
 *  \p numAlgoIds reports how many were written.
 *
 *  @param[in]
 *  lightHandle Pointer to the allocated aclBLASLt handle. See \ref aclblasLtHandle_t.
 *  @param[in]
 *  computeType Compute precision. See \ref aclblasComputeType_t.
 *  @param[in]
 *  scaleType Scale data type. See \ref aclDataType.
 *  @param[in]
 *  Atype,Btype,Ctype,Dtype Data types of the A, B, C, and D matrices.
 *  @param[out]
 *  algoIdsArray Output array receiving the supported algorithm IDs.
 *  @param[in]
 *  algoIdsArrayLength Capacity of \p algoIdsArray (in elements).
 *  @param[out]
 *  numAlgoIds Number of algorithm IDs written into \p algoIdsArray.
 *
 *  \retval ACLBLAS_STATUS_SUCCESS If the query was successful.
 *  \retval ACLBLAS_STATUS_NOT_INITIALIZED If \p lightHandle is NULL.
 *  \retval ACLBLAS_STATUS_INVALID_VALUE If \p algoIdsArray or \p numAlgoIds is
 *  NULL, or \p algoIdsArrayLength is less than or equal to zero.
 *  \retval ACLBLAS_STATUS_NOT_SUPPORTED If the data type combination is
 *  unsupported.
 */
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

/*! \ingroup library_module
 *  \brief Retrieve the possible algorithms.
 *
 *  \details
 *  This function retrieves the possible algorithms for the matrix multiply
 *  operation aclblasLtMatmul() with the given input matrices A, B, and
 *  C, and the output matrix D. The output is placed in ``heuristicResultsArray``
 *  in order of increasing estimated compute time.
 *
 *  @param[in]
 *  lightHandle Pointer to the allocated aclBLASLt handle for the
 *  aclBLASLt context. See \ref aclblasLtHandle_t.
 *  @param[in]
 *  matmulDesc Handle to a previously created matrix multiplication
 *  descriptor of type \ref aclblasLtMatmulDesc_t.
 *  @param[in]
 *  Adesc,Bdesc,Cdesc,Ddesc Handles to the previously created matrix layout
 *  descriptors of the type \ref aclblasLtMatrixLayout_t.
 *  @param[in]
 *  pref Pointer to the structure holding the heuristic
 *  search preferences descriptor. See \ref aclblasLtMatmulPreference_t.
 *  @param[in]
 *  requestedAlgoCount Size of the \p heuristicResultsArray (in elements).
 *  This is the requested maximum number of algorithms to return.
 *  @param[out]
 *  heuristicResultsArray[] Array containing the algorithm heuristics and
 *  associated runtime characteristics returned by this function, in order
 *  of increasing estimated compute time.
 *  @param[out]
 *  returnAlgoCount Number of algorithms returned by this function. This
 *  is the number of \p heuristicResultsArray elements written.
 *
 *  \retval ACLBLAS_STATUS_SUCCESS If query was successful. Inspect
 *  ``heuristicResultsArray[0 to (returnAlgoCount -1)].state`` for the status of the
 *  results.
 *  \retval ACLBLAS_STATUS_INVALID_VALUE If any pointer argument is NULL,
 *  \p requestedAlgoCount is less than or equal to zero, or the compute type is
 *  incompatible with the input matrix data types.
 */
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

// Matmul
/*! \ingroup library_module
 *  \brief Matrix-matrix multiplication.
 *
 *  \details
 *  This function computes the matrix multiplication of matrices A and B to
 *  produce the output matrix D, according to the following operation: \p D = \p
 *  alpha*( \p A *\p B) + \p beta*( \p C ), where \p A, \p B, and \p C are input
 *  matrices, and \p alpha and \p beta are input scalars. Note: This function
 *  supports both in-place matrix multiplication (``C == D`` and ``Cdesc == Ddesc``) and
 *  out-of-place matrix multiplication (``C != D``).
 *
 *  @param[in]
 *  lightHandle Pointer to the allocated aclBLASLt handle for the
 *  aclBLASLt context. See \ref aclblasLtHandle_t.
 *  @param[in]
 *  computeDesc Handle to a previously created matrix multiplication
 *  descriptor of type \ref aclblasLtMatmulDesc_t.
 *  @param[in]
 *  alpha,beta Pointers to the scalars used in the multiplication.
 *  @param[in]
 *  Adesc,Bdesc,Cdesc,Ddesc Handles to the previously created matrix layout
 *  descriptors of the type \ref aclblasLtMatrixLayout_t.
 *  @param[in]
 *  A,B,C Pointers to the memory associated with the
 *  corresponding descriptors \p Adesc, \p Bdesc, and \p Cdesc.
 *  @param[out]
 *  D Pointer to the memory associated with the
 *  descriptor \p Ddesc.
 *  @param[in]
 *  algo Handle for matrix multiplication algorithm to be
 *  used. See \ref aclblasLtMatmulAlgo_t. When NULL, an implicit heuristics query
 *  with default search preferences will be performed to determine the actual
 *  algorithm to use.
 *  @param[in]
 *  workspace Pointer to the workspace buffer allocated in the GPU
 *  memory. Pointer must be 16B aligned.
 *  @param[in]
 *  workspaceSizeInBytes Size of the workspace.
 *  @param[in]
 *  stream The stream where all device work is submitted.
 *
 *  \retval ACLBLAS_STATUS_SUCCESS If the operation completed
 *  successfully.
 *  \retval ACLBLAS_STATUS_NOT_INITIALIZED If the aclBLASLt handle has not been initialized.
 *  \retval ACLBLAS_STATUS_NOT_SUPPORTED If the current implementation on the
 *  selected device doesn't support the configured operation.
 *  \retval ACLBLAS_STATUS_INVALID_VALUE If the parameters are unexpectedly NULL, in
 *  conflict, or in an impossible configuration.
 */
aclblasStatus_t aclblasLtMatmul(aclblasLtHandle_t lightHandle,
                                aclblasLtMatmulDesc_t computeDesc,
                                const void* alpha,
                                const void* A,
                                aclblasLtMatrixLayout_t Adesc,
                                const void* B,
                                aclblasLtMatrixLayout_t Bdesc,
                                const void* beta,
                                const void* C,
                                aclblasLtMatrixLayout_t Cdesc,
                                void* D,
                                aclblasLtMatrixLayout_t Ddesc,
                                const aclblasLtMatmulAlgo_t* algo,
                                void* workspace,
                                size_t workspaceSizeInBytes,
                                aclrtStream stream);

/*! \ingroup matrix_transform_module
 *  \brief Create a matrix transform descriptor.
 *  \param transformDesc Pointer to the descriptor to be created.
 *  \param scaleType Compute (scale) data type. See ``aclDataType``.
 *  \retval ACLBLAS_STATUS_SUCCESS On success.
 *  \retval ACLBLAS_STATUS_INVALID_VALUE If transformDesc is NULL.
 *  \retval ACLBLAS_STATUS_ALLOC_FAILED If the descriptor allocation fails.
 */
aclblasStatus_t aclblasLtMatrixTransformDescCreate(aclblasLtMatrixTransformDesc_t* transformDesc,
                                                   aclDataType scaleType);

/*! \ingroup matrix_transform_module
 *  \brief Destroy a matrix transform descriptor.
 *  \param transformDesc Descriptor to be destroyed.
 *  \retval ACLBLAS_STATUS_SUCCESS On success.
 *  \retval ACLBLAS_STATUS_INVALID_VALUE If transformDesc is NULL.
 */
aclblasStatus_t aclblasLtMatrixTransformDescDestroy(const aclblasLtMatrixTransformDesc_t transformDesc);

/*! \ingroup matrix_transform_module
 *  \brief Set an attribute of a matrix transform descriptor.
 *  \param transformDesc Descriptor to modify.
 *  \param attr Attribute to set. See ``aclblasLtMatrixTransformDescAttribute_t``.
 *  \param buf Source buffer holding the attribute value.
 *  \param sizeInBytes Size of the source buffer in bytes.
 *  \retval ACLBLAS_STATUS_SUCCESS On success.
 *  \retval ACLBLAS_STATUS_INVALID_VALUE If the parameters are NULL or the size mismatches.
 *  \retval ACLBLAS_STATUS_NOT_SUPPORTED If \p attr is not a recognized attribute.
 */
aclblasStatus_t aclblasLtMatrixTransformDescSetAttribute(aclblasLtMatrixTransformDesc_t transformDesc,
                                                         aclblasLtMatrixTransformDescAttribute_t attr,
                                                         const void* buf,
                                                         size_t sizeInBytes);

/*! \ingroup matrix_transform_module
 *  \brief Query an attribute of a matrix transform descriptor.
 *  \param transformDesc Descriptor to query.
 *  \param attr Attribute to query. See ``aclblasLtMatrixTransformDescAttribute_t``.
 *  \param buf Destination buffer for the attribute value.
 *  \param sizeInBytes Size of the destination buffer in bytes.
 *  \param sizeWritten On return, the number of bytes written (or required).
 *  \retval ACLBLAS_STATUS_SUCCESS On success.
 *  \retval ACLBLAS_STATUS_INVALID_VALUE If the parameters are NULL or the buffer is too small.
 *  \retval ACLBLAS_STATUS_NOT_SUPPORTED If \p attr is not a recognized attribute.
 */
aclblasStatus_t aclblasLtMatrixTransformDescGetAttribute(aclblasLtMatrixTransformDesc_t transformDesc,
                                                         aclblasLtMatrixTransformDescAttribute_t attr,
                                                         void* buf,
                                                         size_t sizeInBytes,
                                                         size_t* sizeWritten);

/*! \ingroup matrix_transform_module
 *  \brief Compute C = alpha * op(A) + beta * op(B) with layout / type transform.
 *  \param lightHandle aclBLASLt context handle.
 *  \param transformDesc Matrix transform descriptor.
 *  \param alpha Host pointer to the A scaling factor (interpreted per scaleType).
 *  \param A Device pointer to matrix A.
 *  \param Adesc Layout descriptor of A.
 *  \param beta Host pointer to the B scaling factor (interpreted per scaleType).
 *  \param B Device pointer to matrix B (may be NULL when beta is zero).
 *  \param Bdesc Layout descriptor of B (may be NULL when beta is zero).
 *  \param C Device pointer to the output matrix.
 *  \param Cdesc Layout descriptor of C.
 *  \param stream Execution stream.
 *  \retval ACLBLAS_STATUS_SUCCESS On success.
 *  \retval ACLBLAS_STATUS_NOT_INITIALIZED If the handle is NULL.
 *  \retval ACLBLAS_STATUS_INVALID_VALUE If the parameters are NULL or in conflict.
 *  \retval ACLBLAS_STATUS_NOT_SUPPORTED If the dtype / order / op combination is unsupported.
 *  \retval ACLBLAS_STATUS_INTERNAL_ERROR If an internal launch or tiling error occurs.
 */
aclblasStatus_t aclblasLtMatrixTransform(aclblasLtHandle_t lightHandle,
                                         aclblasLtMatrixTransformDesc_t transformDesc,
                                         const void* alpha,
                                         const void* A,
                                         aclblasLtMatrixLayout_t Adesc,
                                         const void* beta,
                                         const void* B,
                                         aclblasLtMatrixLayout_t Bdesc,
                                         void* C,
                                         aclblasLtMatrixLayout_t Cdesc,
                                         aclrtStream stream);

// ============================================================================
// Logger (experimental)
// ============================================================================

/*! \ingroup library_module
 *  \brief Set the logging output file handle.
 *
 *  \details Experimental. Sets the file handle where log messages are written.
 *  Once registered, the provided file handle must not be closed unless this
 *  function is called again to switch to a different handle. If never called,
 *  log output goes to stdout.
 *
 *  \param[in] file Pointer to an open file with write permission.
 *
 *  \retval ACLBLAS_STATUS_SUCCESS If the logging file was successfully set.
 *  \retval ACLBLAS_STATUS_INVALID_VALUE If \p file is NULL.
 */
aclblasStatus_t aclblasLtLoggerSetFile(FILE* file);

/*! \ingroup library_module
 *  \brief Set the logging level.
 *
 *  \details Experimental. Sets the logging level. Messages at or below the
 *  specified level are emitted (subject to the mask). See \ref aclblasLtLogLevel_t.
 *
 *  \param[in] level Logging level. Must be in [0, 5].
 *
 *  \retval ACLBLAS_STATUS_SUCCESS If the logging level was successfully set.
 *  \retval ACLBLAS_STATUS_INVALID_VALUE If \p level is not a valid logging level.
 */
aclblasStatus_t aclblasLtLoggerSetLevel(int level);

/*! \ingroup library_module
 *  \brief Set the logging mask.
 *
 *  \details Experimental. Sets the logging mask as a bitmask of
 *  aclblasLtLogMask_t flags. Only message categories whose bit is set are
 *  emitted. See \ref aclblasLtLogMask_t.
 *
 *  \param[in] mask Logging mask (bitwise OR of ACLBLASLT_LOG_MASK_* values).
 *
 *  \retval ACLBLAS_STATUS_SUCCESS If the logging mask was successfully set.
 */
aclblasStatus_t aclblasLtLoggerSetMask(int mask);

/*! \ingroup library_module
 *  \brief Force-disable logging for the entire run.
 *
 *  \details Experimental. Disables all logging unconditionally. Once called,
 *  logging cannot be re-enabled by any other logger API or environment variable
 *  for the remainder of the process lifetime.
 *
 *  \retval ACLBLAS_STATUS_SUCCESS If logging was successfully disabled.
 */
aclblasStatus_t aclblasLtLoggerForceDisable(void);

/*! \ingroup library_module
 *  \brief Set the logging callback function.
 *
 *  \details Experimental. When set, log messages are delivered to the callback
 *  instead of file/stdout output. See \ref
 *  aclblasLtLoggerCallback_t.
 *
 *  \param[in] callback Pointer to the callback function. Pass NULL to clear.
 *
 *  \retval ACLBLAS_STATUS_SUCCESS If the callback was successfully set.
 */
aclblasStatus_t aclblasLtLoggerSetCallback(aclblasLtLoggerCallback_t callback);

/*! \ingroup library_module
 *  \brief Open and set a logging output file by path.
 *
 *  \details Experimental. Opens the file at \p logFile for writing and sets it
 *  as the logging output. The file handle is managed internally and will be
 *  closed when the logger is reconfigured or at program exit.
 *
 *  \param[in] logFile Path to the logging output file.
 *
 *  \retval ACLBLAS_STATUS_SUCCESS If the logging file was successfully opened.
 *  \retval ACLBLAS_STATUS_INVALID_VALUE If \p logFile is NULL.
 *  \retval ACLBLAS_STATUS_INTERNAL_ERROR If the file cannot be opened.
 */
aclblasStatus_t aclblasLtLoggerOpenFile(const char* logFile);

#ifdef __cplusplus
}
#endif

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * Single device translation unit for the arch22 Cher2k kernels.
 *
 * The device entry points, layout helpers, MMAD machinery and epilogues are
 * split by their dataflow roles below. They are compiled together as one translation
 * unit, exactly like a single monolithic cher2k_kernel.cpp would be, which is
 * required because the launchers and the kernels they queue must share the same
 * template instantiations. System includes and namespace imports belong to this
 * translation unit; the included
 * implementation fragments contain neither.
 */
#include "common/helper/cube_load_params_arch22.h"
#include <cstdint>

#include "kernel_operator.h"
#include "common/helper/scaled_scalar_arch22.h"
#include "common/helper/triangle_store_arch22.h"
#include "cann_ops_blas_common.h"
#include "lib/matmul_intf.h"
#include "cher2k_kernel.h"

using AscendC::BinaryRepeatParams;
using AscendC::CacheLine;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;
using AscendC::DataCopyExtParams;
using AscendC::DataCopyPadExtParams;
using AscendC::DataCopyParams;
using AscendC::FixpipeParamsV220;
using AscendC::GetBlockIdx;
using AscendC::GetBlockNum;
using AscendC::GetMatmulApiTiling;
using AscendC::GetSubBlockIdx;
using AscendC::GlobalTensor;
using AscendC::HardEvent;
using AscendC::HF32Mode;
using AscendC::LoadData2DParams;
using AscendC::LoadData3DParamsV2;
using AscendC::LocalTensor;
using AscendC::MatmulImpl;
using AscendC::MatmulType;
using AscendC::MmadParams;
using AscendC::Nd2NzParams;
using AscendC::PipeBarrier;
using AscendC::ResetMask;
using AscendC::SetFlag;
using AscendC::SyncAll;
using AscendC::TBuf;
using AscendC::TPipe;
using AscendC::TPosition;
using AscendC::TQue;
using AscendC::WaitFlag;

#include "cher2k_kernel_layout.cpp"
#include "cher2k_kernel_cube.cpp"
#include "cher2k_kernel_postprocess.cpp"
#include "cher2k_kernel_entry.cpp"

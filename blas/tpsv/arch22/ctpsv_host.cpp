/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file ctpsv_host.cpp
 * \brief aclblasCtpsv host 层（arch22 / DAV_2201）：参数校验 + tiling 组装 + 单 kernel 直调。
 *
 * Design Contract（DESIGN.md §8）硬约束：
 *  - device 指针 + 异步语义：AP/x 为 Device 内存指针直达 kernel；host 无数据 H2D/D2H、
 *    无内部 aclrtSynchronizeStream；读回前由调用方同步 stream（spec notes 异步执行条）。
 *  - 单 kernel <<<useCoreNum, nullptr, stream>>> 直调（numBlocks 动态传入禁写死），
 *    立即返回；host 仅元数据 tiling H2D 一次（TilingDataDevice GM 传递，DESIGN §3.3/3.4）。
 *  - 本文件为 include/cann_ops_blas.h 新增 aclblasCtpsv 声明的实现（任务书 §2.3 签名，
 *    与 aclblasStpsv 逐参数对应、元素类型 aclblasComplex）。
 *  - 复数纪律：OP_T（纯转置）与 OP_C（共轭转置）是不同算子，host 侧禁止折叠
 *    （CP1 裁定第 7 条 / spec op_c_conjugate_transpose_identity）。
 *
 * tiling 缓冲生命周期（免同步，stpsv/ctrsv 先例）：tiling GM 缓冲按 stream 维度进程级
 * 缓存复用，aclrtMemcpyAsync 与 kernel 同 stream 下发——同 stream 任务天然按序，无竞态、
 * 无阻塞、无泄漏（每 stream 一个缓冲）。内容去重：同 stream 连续调用 tiling 逐字节相同
 * 时跳过 H2D。锁覆盖「去重判定 + H2D + kernel 下发」全序（TOCTOU 防护，ctrsv 先例）。
 *
 * 参数校验顺序（DESIGN §3.4 / spec shape_constraints 锁定，①→⑦ 单违例触发即返回）：
 *   ① handle==nullptr → ACLBLAS_STATUS_HANDLE_IS_NULLPTR（最优先）
 *   ② n<0 → ACLBLAS_STATUS_INVALID_VALUE
 *   ③ n==0 先行 quick return SUCCESS（不引用 AP、不更新 x；先于 ④⑤⑥⑦——n=0 时
 *      AP/x 允许 nullptr，Netlib ctpsv IF (N.EQ.0) RETURN 语义）
 *   ④ uplo/trans/diag 非法枚举 → ACLBLAS_STATUS_INVALID_VALUE（枚举字面量以上游
 *      头文件实态为准：UPPER=121/LOWER=122、OP_N/T/C=111/112/113、NON_UNIT=131/UNIT=132）
 *   ⑤ incx==0 → ACLBLAS_STATUS_INVALID_VALUE
 *   ⑤b incx==INT_MIN → ACLBLAS_STATUS_INVALID_VALUE（-incx 有符号取反未定义行为防护，
 *      ctrsv host 同款拒绝）
 *   ⑥ n>0 且 AP/x==nullptr → ACLBLAS_STATUS_INVALID_VALUE
 *   ⑦ n>65535 → ACLBLAS_STATUS_INVALID_VALUE（工程性索引域防护：uint32 列基址
 *      n(n+1)/2 < 2^31，DESIGN §3.9 论证；支持域 n ≤ 65535 随接口文档标注）
 *   ⑧ x 物理跨度守卫 → ACLBLAS_STATUS_NOT_SUPPORTED：2·span > UINT32_MAX（span =
 *      1+(n-1)·|incx| 复数元素）时 kernel Init 的 (uint32_t)(2ULL·span) x GM 视图长度
 *      截断回绕 → 静默越界，host 前置拒绝（stpsv span 守卫先例，n=4/incx=1431655765
 *      恰好回绕边界）
 *
 * 迭代二（PLAN.md 迭代二，stpsv/ctrsv 迭代二先例）：12 编译期模板全量实例化
 * （OP_C 独立分支不折叠）+ incx≠1/负步长路径（kernel 侧 FAST/GATHER/STRIDED 运行期
 * 选择），迭代一 NOT_SUPPORTED 门控（7 项：其余 11 模式 / incx≠1 / n>4096）全部移除。
 * 替代物 = kernel 能力域守卫（host/kernel 同式 UB 联合不等式，ctrsv span8 守卫先例）：
 * bR/bI 双平面常驻 + 装载暂存 + 求解器缓冲最小足迹（STRIDED 口径）超 184KB 向量 UB
 * 预算 → NOT_SUPPORTED——大 n（约 >10K）x 平面分段驻留属迭代三「规格完整」scope
 *（DESIGN §3.9；测试覆盖口径 n ≤ 4096 不受影响）。
 */

#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "common/helper/aclblas_handle_internal.h"
#include "ctpsv_tiling_data.h"

// 单模板单 kernel 直调入口（uplo→trans→diag 三层 switch 在 kernel 侧；迭代二 12 模板
// 全量实例化，12 分支全通——OP_C 独立分支不折叠）
void ctpsv_kernel_do(
    uint8_t* AP, uint8_t* x, uint8_t* tilingGm, aclblasFillMode_t uplo, aclblasOperation_t trans,
    aclblasDiagType_t diag, uint32_t numBlocks, void* stream);

// opt2 P2：小 n（≤8）标量快路径直调入口（n≤CTPSV_SMALL_N_MAX 且 incx=1 时由
// aclblasCtpsv 直调；guard 谓词与 UT 镜像同源，结构隔离定谳见 ctpsv_kernel.cpp）
// opt3 E1：n 作 launch 标量参数直传（签名第 3 参；_small 全程零 tiling 读，
// tilingGm 参数移除——kernel 侧不再消费 tiling）。
void ctpsv_kernel_do_small(
    uint8_t* AP, uint8_t* x, int n, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag,
    uint32_t numBlocks, void* stream);

// per-stream tiling GM 缓冲池（含内容去重）：免同步生命周期管理（见文件头注释；stpsv 先例）
struct CtpsvTilingSlot {
    uint8_t* dev = nullptr;
    CtpsvTilingDataDevice last = {};
    bool valid = false;
    std::mutex mu; // 序列化同 stream 并发调用的去重读改写（跨 stream 由池锁隔离）
};

static CtpsvTilingSlot* CtpsvGetTilingSlot(aclrtStream stream)
{
    static std::mutex poolMu;
    static std::map<aclrtStream, CtpsvTilingSlot> pool;
    std::lock_guard<std::mutex> lock(poolMu);
    auto it = pool.find(stream);
    if (it != pool.end()) {
        return &it->second;
    }
    // 槽内含 std::mutex（不可拷贝）——try_emplace 原位默认构造，dev 事后回填
    auto ins = pool.try_emplace(stream);
    uint8_t* buf = nullptr;
    if (aclrtMalloc(reinterpret_cast<void**>(&buf), sizeof(CtpsvTilingDataDevice), ACL_MEM_MALLOC_HUGE_FIRST) !=
        ACL_SUCCESS) {
        pool.erase(ins.first);
        return nullptr;
    }
    ins.first->second.dev = buf;
    return &ins.first->second;
}

static CtpsvTilingDataDevice CalCtpsvTilingData(int n, int incx, const aclblasComplex* AP, aclblasComplex* x)
{
    CtpsvTilingDataDevice tiling = {};
    tiling.n = static_cast<uint32_t>(n);
    tiling.incx = static_cast<int32_t>(incx);
    tiling.apPtr = reinterpret_cast<uint64_t>(AP);
    tiling.xPtr = reinterpret_cast<uint64_t>(x);
    tiling.useCoreNum = 1; // 串行依赖链单 AIV 核闭合（多核列切分为后备杠杆，DESIGN §4.1）
    // AP 列段分块宽度（迭代三 probe_chunk_ab 定谳，遗留决策点 2 关闭）：按 n 自适应——
    //   incx=1（FAST 路径）：n ≤ 4096 取 chunk = n（整列单块；2048<n≤4096 档 n=4096
    //     实测快 11.5%，B/A=0.885 两轮 ×3 A/B/A 复现）；n > 4096 收缩 1024（§3.9 联动）。
    //     ⚠ UB 门控（CtpsvKernelUbFeasible 同步整列口径）：整列单块最重档 n=4096/V2
    //     = 183,360B ≤ 184KB 恰过、余量 ~5KB——kernel 侧列段双 slot（dbuf）对该档
    //     自动关闭（+32KB 超预算），FAST 整列档禁止再叠加 buffer。
    //   incx≠1（GATHER/STRIDED 路径）：维持 min(n,2048)——整列单块在 n=3072/incx=2
    //     实证击穿 GATHER 预算 205.9KB（probe_chunk_ab）。
    if (n > 4096) {
        tiling.chunk = 1024U;
    } else if (incx == 1) {
        tiling.chunk = static_cast<uint32_t>(n);
    } else {
        tiling.chunk = (static_cast<uint32_t>(n) < 2048U) ? static_cast<uint32_t>(n) : 2048U;
    }
    // |incx| 预计算（0 = incx=1 快路径标记，DESIGN §3.3）；迭代一 kernel 仅消费 incx=1。
    // int64 中间量取绝对值（防御性写法：INT_MIN 已被 Validate ⑤b 拒绝，-incx 无溢出 UB）
    tiling.negInc =
        (incx == 1) ? 0U : static_cast<uint32_t>(incx < 0 ? -static_cast<int64_t>(incx) : static_cast<int64_t>(incx));
    tiling.reserved = 0;
    return tiling;
}

// ── 迭代二 kernel 能力域守卫（host/kernel 同式 UB 联合不等式，ctrsv span8 守卫先例）──
//   最小足迹 = STRIDED 口径（GATHER 由 kernel 侧按 image+偏移表不等式择优，超界自动
//   回退 STRIDED；STRIDED 无跨度装载像/偏移表，任意跨度功能正确）：
//     incx=1  ：bufB(2·align64(n)) + bufXTrans(align64(2·ceil8(n))) + 求解器缓冲
//     incx≠1  ：bufB(2·align64(n)) + bufEl(8) + 求解器缓冲
//   求解器缓冲（与 kernel Init 同式）：axpy = col + aR/aI + diag + off；
//   dot 另加 tR/tI/t2 + w/w2。col 项按 ride 谓词条件化（opt2 修复
//   2026-09-27，issue_20260926_opt2-ub-guard-domain-shrink_01 方向 A）：搭车模板
//   （LOWER 全档 / UPPER·T/C·NON_UNIT，kernel kColRideFloats 同式判定）col 项计数
//   常驻精确需求 = align64(2·chunkPad) + 16（搭车前缀 ≤8 元素/16 float + vred src
//   偏移 16 float；对齐垫 48 float 为 64-float 对齐越读护栏，不进常驻预算——kernel
//   实分配维持 align64(2·chunkPad+64)，守卫口径 实分配 ≤ 守卫 + 48 float，由
//   +2048B 余量吸收，恢复带最大实分配 186,432B 仍 ≤184KB）；非搭车模板 = 原布局
//   align64(2·chunkPad)。超 184KB（向量可用 UB 估算）→ NOT_SUPPORTED。
//   支持域边界 = 迭代三原值：N 10528 / T·C 9632（incx=1）、21056 / 19264（incx≠1）。
static bool CtpsvKernelUbFeasible(
    aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n, int incx)
{
    const int64_t v = n;
    // chunk 策略与 CalCtpsvTilingData 同式（迭代三 probe_chunk_ab 定谳：FAST 整列单块
    // 自适应；host/kernel 守卫同式纪律不变）
    const int64_t chunk = (n > 4096) ? 1024 : ((incx == 1) ? v : ((v < 2048) ? v : 2048));
    const auto ceil8 = [](int64_t x) { return (x + 7) / 8 * 8; };
    const auto align64 = [](int64_t x) { return (x + 63) / 64 * 64; };
    const int64_t chunkPad = ceil8(chunk);
    // ride 谓词与 kernel kColRideFloats 条件同式（搭车模板 = LOWER / UPPER·T/C·NON_UNIT）
    const int64_t ride = ((uplo == ACLBLAS_LOWER) || (trans != ACLBLAS_OP_N && diag == ACLBLAS_NON_UNIT)) ? 16 : 0;
    // opt4 L3（2026-09-28）：LOWER 头段窗口化 plane 增量（kernel planeRide 同式）——
    //   多 chunk 档（chunk<n：GATHER n>2048 / FAST n>4096）merged chunk0 消费长
    //   ml+8 超 chunkPad，aR/aI（V1）/ tR/tI/t2（V2）各 +64 float 对齐界；整列单块档
    //   （chunk==n，含全部 n≤4096 FAST 域）len'=n-lb ≤ n 恒不溢出 = 0（UB 布局不变）。
    //   支持域边界缩移（LOWER 多 chunk 档）：incx=1 N 10528→10496 / T·C 9632→9536；
    //   UPPER 模板不变（10528/9632）；incx≠1 LOWER 同步缩移。测试覆盖域 n≤5000 不受影响。
    const int64_t planeRide = ((uplo == ACLBLAS_LOWER) && (chunk < v)) ? 64 : 0;
    const int64_t solver = align64(2 * chunkPad) + ride + 2 * (align64(chunkPad) + planeRide) + 16 + 256 +
                           ((trans != ACLBLAS_OP_N) ? (3 * (align64(chunkPad) + planeRide) + 2 * 256) : 0);
    const int64_t floats = (incx == 1) ? 2 * align64(v) + align64(2 * ceil8(v)) + solver : 2 * align64(v) + 8 + solver;
    return floats * 4 + 2048 <= 184LL * 1024LL;
}

// 参数校验 ④→⑦（DESIGN §3.4 顺序锁定，单违例触发即返回；返回 false = 违例）。
// ①②③ 由 aclblasCtpsv 先行承载：③ n==0 quick return SUCCESS 必须先于本链——n=0 时
// AP/x 允许 nullptr 且不引用枚举（Netlib ctpsv IF (N.EQ.0) RETURN 语义）。
static inline bool ValidateCtpsvArgs(
    aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n, const aclblasComplex* AP,
    const aclblasComplex* x, int incx)
{
    if ((uplo != ACLBLAS_UPPER) && (uplo != ACLBLAS_LOWER)) { // ④ 非法枚举
        return false;
    }
    if ((trans != ACLBLAS_OP_N) && (trans != ACLBLAS_OP_T) && (trans != ACLBLAS_OP_C)) {
        return false;
    }
    if ((diag != ACLBLAS_NON_UNIT) && (diag != ACLBLAS_UNIT)) {
        return false;
    }
    if (incx == 0) { // ⑤
        return false;
    }
    if (incx == std::numeric_limits<int>::min()) { // ⑤b：-incx 有符号取反 UB 防护（ctrsv 先例）
        return false;
    }
    if ((AP == nullptr) || (x == nullptr)) { // ⑥（n>0 时）
        return false;
    }
    if (n > 65535) { // ⑦ 索引域防护（DESIGN §3.9：n(n+1)/2 < 2^31 ⇒ n ≤ 65535）
        return false;
    }
    return true;
}

// ⑧ x 物理跨度 uint32 域守卫（stpsv span 守卫先例，校验序 ⑧ 在 aclblasCtpsv 承载）：
//   kernel Init 以 (uint32_t)(2ULL·span) 设 x GM 视图长度（span = 1+(n-1)·|incx| 复数
//   元素，每元素 2 float；int64 中间积防上溢）。极大 |incx| 时 2·span 截断回绕
//   （n=4/incx=1431655765 → span=4294967296，2·span 回绕 0）→ GM 视图失真静默越界，
//   host 前置拒绝。语义归属能力域（stpsv 同款）：ACLBLAS_STATUS_NOT_SUPPORTED。
static bool CtpsvSpanFeasible(int n, int incx)
{
    const int64_t absIncx = (incx >= 0) ? static_cast<int64_t>(incx) : -static_cast<int64_t>(incx);
    const int64_t span = (n <= 1) ? 1 : 1 + static_cast<int64_t>(n - 1) * absIncx;
    return 2LL * span <= static_cast<int64_t>(UINT32_MAX);
}

// tiling 槽 H2D 去重 + kernel 下发（含小 n 标量快路径分流）。锁覆盖「去重判定 + H2D +
// kernel 下发」全序（TOCTOU 防护，stpsv/ctrsv 先例）；异步返回，无内部同步。
static aclblasStatus_t CtpsvLaunchKernel(
    const CtpsvTilingDataDevice& tiling, const aclblasComplex* AP, aclblasComplex* x, int n, int incx,
    aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, aclrtStream useStream)
{
    CtpsvTilingSlot* tilingSlot = CtpsvGetTilingSlot(useStream);
    if (tilingSlot == nullptr || tilingSlot->dev == nullptr) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    std::lock_guard<std::mutex> lock(tilingSlot->mu);
    if (!tilingSlot->valid || std::memcmp(&tilingSlot->last, &tiling, sizeof(CtpsvTilingDataDevice)) != 0) {
        if (aclrtMemcpyAsync(
                tilingSlot->dev, sizeof(CtpsvTilingDataDevice), &tiling, sizeof(CtpsvTilingDataDevice),
                ACL_MEMCPY_HOST_TO_DEVICE, useStream) != ACL_SUCCESS) {
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
        tilingSlot->last = tiling;
        tilingSlot->valid = true;
    }
    // 小 n（≤8）且 incx=1 → 标量快路径独立 kernel（_small 分离符号，主路径 kernel
    // codegen 零扰动；谓词 = CTPSV_SMALL_N_MAX 共享真源同源）
    if ((n <= static_cast<int>(CTPSV_SMALL_N_MAX)) && (incx == 1)) {
        ctpsv_kernel_do_small(
            reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(AP)), reinterpret_cast<uint8_t*>(x), n, uplo, trans,
            diag, tiling.useCoreNum, useStream);
    } else {
        ctpsv_kernel_do(
            reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(AP)), reinterpret_cast<uint8_t*>(x), tilingSlot->dev,
            uplo, trans, diag, tiling.useCoreNum, useStream);
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasCtpsv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n,
    const aclblasComplex* AP, aclblasComplex* x, int incx)
{
    // ── 参数校验（DESIGN §3.4 顺序锁定 ①→⑦，单违例触发即返回）──
    if (handle == nullptr) { // ①
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (n < 0) { // ②
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) { // ③ quick return：先于枚举/incx/空指针校验（不引用 AP、不更新 x）
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (!ValidateCtpsvArgs(uplo, trans, diag, n, AP, x, incx)) { // ④⑤⑥⑦
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    // ⑧ x 物理跨度 uint32 截断守卫（stpsv span 守卫先例；kernel Init 同式 span 计算的
    //    host 镜像——kernel 侧 span64 已用 int64 中间积，仅 GM 视图长度截断点需 host 拦）
    if (!CtpsvSpanFeasible(n, incx)) {
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }

    // ── 迭代二 kernel 能力域守卫（迭代一 7 项 NOT_SUPPORTED 门控移除后的替代物）──
    //   12 模式 × 任意非零 incx（含负步长）× n ≤ 4096 全通；超 UB 预算的大 n 档
    //   （x 平面分段驻留，DESIGN §3.9 迭代三「规格完整」scope）暂返 NOT_SUPPORTED。
    if (!CtpsvKernelUbFeasible(uplo, trans, diag, n, incx)) {
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }

    aclrtContext currentCtx = nullptr;
    if (aclrtGetCurrentContext(&currentCtx) != ACL_SUCCESS || currentCtx == nullptr) {
        return ACLBLAS_STATUS_NOT_INITIALIZED;
    }
    auto* h = reinterpret_cast<_aclblas_handle*>(handle);
    aclrtStream useStream = h->stream;

    CtpsvTilingDataDevice tiling = CalCtpsvTilingData(n, incx, AP, x);

    // 异步返回：无数据 H2D/D2H、无内部同步；调用方读回前自行 aclrtSynchronizeStream
    return CtpsvLaunchKernel(tiling, AP, x, n, incx, uplo, trans, diag, useStream);
}

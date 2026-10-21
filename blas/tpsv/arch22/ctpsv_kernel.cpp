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
 * \file ctpsv_kernel.cpp
 * \brief aclblasCtpsv AIV 求解内核（arch22 / DAV_2201）。
 *
 * 迭代二（DESIGN.md §7 / PLAN.md 迭代二）：12 编译期模板全量实例化
 * （uplo(2) × trans{N,T,C}(3) × diag(2)，OP_C 独立分支不折叠，CP1 裁定第 7 条）：
 *   形态 V1 右视列 axpy（trans=OP_N）：
 *     T7 <LOWER,N,NON_UNIT> / T8 <LOWER,N,UNIT>      前代 i:0→n-1（迭代一主路径）
 *     T1 <UPPER,N,NON_UNIT> / T2 <UPPER,N,UNIT>      回代 i:n-1→0（trailing 前缀 [0,j) 起点对齐）
 *   形态 V2 左视列 dot（trans=OP_T/OP_C；OP_T 不可 axpy 化——packed 下 A^T 列为跨列
 *   散取不连续，dot 为定案形态，DESIGN §3.1）：
 *     T3-T6 <UPPER,T/C,·>                            前代 i:0→n-1（dot 区 = 行 [0,i)）
 *     T9-T12 <LOWER,T/C,·>                           回代 i:n-1→0（dot 区 = 行 (i,n)）
 *   OP_C 共轭三处承载（DESIGN §3.5）：①dot 系数共轭 = 符号模式编译期融合（迭代三切换，
 *   见下）；②头段标量 dot 符号模式编译期分支；③对角复除 invIm 符号位编译期翻转（除
 *   conj(d)）；diag=UNIT 对角零读取（假定 1.0+0.0i，免除复数除法）。
 *
 * 迭代三性能优化（PLAN.md iterations[3]；穿刺情报 PROBE_SUMMARY 2026-09-26）：
 *   1) probe_opc_conj 情报落地：OP_C dot 系数共轭由「aI 平面 Muls(-1) 一步 + OP_T 统一
 *      公式」切换为「符号模式编译期融合」——tR = aR·bR + aI·bI、tI = aR·bI − aI·bR
 *      （conj(a)·b 展开；穿刺 72/72 实测比 Muls 形态快 4.1-4.7% 且位级一致）。列段
 *      装载路径不再发射 Muls(-1)（省一条全宽向量指令/chunk）。
 *   2) probe_neg_incx 情报落地：AP 列段 GM→UB 装载由「每 chunk 串行 DataCopyPad +
 *      MTE2→V 全量同步」改为「chunk 级 look-ahead 预取流水」（DESIGN §4.3 优化杠杆
 *      序列之「双缓冲深度 / look-ahead」）：消费当前 chunk 前先发射全局后继 chunk
 *      （同步下一 chunk 或跨步 look-ahead 到后续步骤首 chunk），MTE2 装载与 V 单元
 *      计算重叠；col 缓冲双 slot 轮转（dbuf，UB 预算内启用；超预算档自动退回单 slot
 *      串行，足迹与迭代二守卫式一致）。同步事件独立编号（MTE2_V_PF=3），与对角/头段
 *      微拷贝的 MTE2_V(2) 交替 set/wait 不互扰（事件 id 写死顺序复用纪律）。
 *      附带：装载路径的 per-chunk Duplicate 全宽清零移除——vred 越界读残片的消费域
 *      已由「vred 后 [len,lenA) pad lane 显式清零 + 计数形式 Mul/Axpy 只消费 lenA lane
 *      + RRS 级联前 [lenA,lenR) 尾 lane 显式清零」三重护栏封闭（迭代二 dot 级联尾
 *      lane 修复同域论证），全宽清零为冗余 V 带宽（2·chunkPad float/chunk）。
 *   3) 头段（≤7 复数元素）微拷贝移驻 bufDiag（原 colIl slot0）：头段仅 MTE2 写 + 标量
 *      读（V 不触碰 bufDiag），MTE2 按序 + 标量程序序即安全（原 DrainV 为保护 colIl
 *      的 V 读，移驻后不适用）；同时消除「跨步预取与头段装载争用 colIl slot0」的
 *      覆写冲突（预取数据在被消费前被头段覆写）。
 *
 * 迭代四 opt2 P0（OPT2_PLAN P0；DESIGN §3.5 装载策略变更，2026-09-26 用户批准）：
 *   头段/对角微拷贝搭车列段 bulk 装载（msprof 画像定谳：微拷贝占 MTE2 操作数 47%，
 *   墙钟占比 8.6~10.2%；单位成本 NON_UNIT−UNIT 配对锚定 0.13~0.15µs/次）：
 *   1) LOWER chunk0 装载起点 8 对齐回退 L（NON_UNIT：L=step&~7，前缀含对角；UNIT：
 *      L=(step+1)&~7，对齐步零前缀），前缀 ≤8 元素 + 头段并入单次 DataCopyPad；主段
 *      自装载内元素 8（float 偏移 16，32B 对齐——vred src 硬门）消费，头段/对角由
 *      col slot 标量直读（MTE2_V_PF wait 覆盖标量读，LoadX STRIDED 先例；stpsv
 *      ComputeMainFast d=buf.GetValue(dot) 同构）——消每步 1~2 次微拷贝 DMA +
 *      MTE2_V(2) 事件对。算术语序与迭代三逐位一致：头段标量累加序 / 主段 lane 映射 /
 *      RRS 级联分组均不变（bit-exact 保持，多 chunk 专项域安全）。
 *      前缀垃圾 lane [L,step)（含对角 lane）不经 vred 消费域（src 偏移 16 起），
 *      无 0·Inf=NaN 暴露面。
 *   2) UPPER 左视（T3/T5）末块装载长 +1（对角 = 装载末元素，stpsv FastLoadLen=dot+1
 *      同构）；对角 lane 落入 [len,lenA) 时由既有 pad 清零护栏覆盖（标量读出后清零）。
 *   3) 时序不可达豁免：V1 UPPER（T1）对角在末块、步骤首即需（解 x_j 先于 trailing）
 *      ——维持 LoadDiagScalar 独立微拷贝；主段空（cnt=0）尾步（每轮 ≤8 步）走遗留
 *      bufDiag 微拷贝路径。
 *   4) col slot 宽度按模板条件增长 kColRideFloats（LOWER 与 UPPER V2 NON_UNIT = +64
 *      float：搭车前缀 ≤8 元素/16 float + vred src 偏移 16 float 后的 64-float 对齐越读
 *      护栏；T1/T2/T4/T6 无搭车 = 0，UB 布局与迭代三逐字节一致——T2 档实测 col 布局
 *      漂移致 +84µs scalar 税（msprof 定谳），无搭车档回退原布局规避）。
 *      opt2 修复（2026-09-27，issue_20260926_opt2-ub-guard-domain-shrink_01 方向 A）：
 *      kernel 侧分配维持本式不变（dbuf 双 slot 与单 slot 同宽 +64 对齐界，slot 起址
 *      256B 对齐不破，代码生成与 opt2 终态逐字节一致）；host 守卫 CtpsvKernelUbFeasible
 *      col 项按 ride 谓词条件化计数常驻精确需求 = align64 基 + 16 float（搭车前缀
 *      16 float + vred src 偏移 16 float；对齐垫 48 float 为越读护栏不进常驻预算，
 *      由守卫 +2048B 余量吸收——守卫口径 kernel 实分配 ≤ 守卫 + 48 float，恢复带
 *      最大实分配 186,432B 仍 ≤184KB 预算），支持域边界恢复迭代三原值
 *      （N 10528 / T·C 9632 / incx≠1 21056·19264）。
 *
 * 迭代四 opt2 P1（OPT2_PLAN P1；DESIGN §3.5 变更记录，2026-09-26 opt2 战役用户已批）：
 *   dot 尾清零标量环向量化——DotChunk 级联前尾 lane 零垫 [lenA,lenR)×{tR,tI} 由
 *   ~56 次/步标量 SetValue 环改为单条 Duplicate/平面（stpsv IssueMainFast「pad 零垫
 *   收缩」同域先例：窄域 ≤56 lane、count 非 64 倍数、dst 子偏移 8-float/32B 对齐——
 *   同架构真机交付形态）。位级等价（写域/写值不变）；tR/tI 为 V 专属缓冲，与在飞
 *   col 预取装载区不相交，无涉 V_MTE2 事件对；仅 V2 模板（T3-T6/T9-T12）DotChunk
 *   实例化路径触及，V1（T1/T2/T7/T8）不调用 DotChunk——控制序/装载算式逐位保持
 *   （v6 教训：未改动模板等价重排亦是变量，A/B 逐模板核对中性）。
 *
 * 迭代四 opt2 P2（OPT2_PLAN P2；DESIGN §3.5 变更记录，2026-09-26 opt2 战役二）：
 *   小 n（≤8）FAST 标量快路径（独立入口分发，见 CtpsvSmallSolve）——opt2_smalln
 *   诊断定谳（msprof + 空事件窗分解）：小 n 地板带（TC_PF_1005-1008，全
 *   LOWER/N/NON_UNIT/incx=1 → T7）的主体是 kernel 自身确定性执行时间（n=1 Task
 *   Duration 4.4-6.4µs、scalar busy 2.98µs），其中 ①BuildInterleaveOffsets 256 次
 *   SetValue ≈0.8µs 为 n 无关固定项（n=1 也全额支付）；②n≤8 时 StepSegStart 钳位
 *   n → 全程零 col chunk，逐步退化为 LoadDiagScalar + ApplyHeadMac 各一组 MTE2
 *   微拷贝事件对（n=8 共 ~23 组事件对 ≈2.4µs）。快路径：x 与 packed AP 各一次
 *   DataCopyPad 整段入 UB（AP n(n+1)/2 复数 ≤36 元素 288B，独立 TBuf），纯标量
 *   求解（Netlib 镜像语序 = CPU golden 方言：V1 LOWER/UPPER 外内循环方向、V2 dot
 *   累加序 UPPER 升序/LOWER 降序均与 golden 逐位同序），一次 DataCopyPad 写回；
 *   事件对 7+2/步 → 3 组。guard = incx==1 && n≤CTPSV_SMALL_N_MAX（host 侧
 *   aclblasCtpsv 求值直调 ctpsv_kernel_do_small；12 模板 constexpr 全覆盖——V1
 *   axpy / V2 dot(OP_T/OP_C 符号融合) / UNIT 直通均与主路径同语义）；CtpsvKernel
 *   类体与主入口宏逐位保持 P1（结构隔离三连教训：①内联进类 +3.65%、②入口宏
 *   tiling 预读分支 +3.2%（均 T7 档 n>8 未改路径 scalar 税，msprof TC_PF_1111
 *   1422.8→1474.8/1468.2µs 定谳，P1 同会话复测 1420.5 排除环境漂移）→ 终态 =
 *   分离 _small kernel 符号 + host 分发，主路径 codegen 零扰动；v6「未改动模板
 *   等价重排亦是变量」教训的代码布局变体）。
 *   精度：标量语序即 golden 语序（fp32 逐位同序 MAC + 复除同式），双门容差
 *   天然对齐；n>8 主路径逐位不动。
 *
 * 迭代四 opt4 L3（OPT4_PLAN L3；2026-09-28，非位级杆——容差域验证）：
 *   LOWER 头段窗口化（LOWER 全 6 模板受益；P0 画像定谳头段税 V1 +118 / V2 +145 ns/step
 *   全在 scalar pipe）。merged chunk0 的 vred 消费由「srcOff=16 跳过前缀」改为
 *   「srcOff=0 覆盖前缀+主段（len' = ml + pref，pref ∈ {0,8}）」，头段元素 [i+1, mb)
 *   并入向量主计算（V1 Axpy / V2 dot），头段标量环（~35 op + 1 barrier/步）移除；
 *   垃圾 lane [lb, i]（前列尾数据 + 对角 lane，maskLen = s0−lb ≤ 8）掩蔽：
 *     V1 = ConsumeColChunkMerged 内 aR/aI [0,maskLen) 前置 Duplicate 清零（Axpy 直写
 *          b 平面，b ± 0 保 NaN/Inf 语义；对角 lane b_j=x_j 经 S_V 对先行落值）；
 *     V2 = DotChunk 内 tR/tI [0,maskLen) 乘加合成后掩蔽（前置 a 清零仍有 0×Inf=NaN
 *          暴露——乘法不清 NaN；掩蔽写域与 [lenA,lenR) 尾垫不相交）。
 *   Axpy/DotChunk 基回退 lb、长 +pref；非对齐算术语序变化（头段并入级联/双趟 axpy）
 *   → 非位级等价（双门容差 + 确定性验证）；UNIT 对角 lane 恒在掩蔽域。
 *   plane 增量 planeRide（运行期）：仅多 chunk 档（chunkLen<n：GATHER n>2048 / FAST
 *   n>4096）aR/aI/tR/tI/t2 各 +64 float（len' ≤ chunkPad+8 → ceil64 需 +64 对齐界）；
 *   整列单块档（chunkLen==n，含全部 n≤4096 FAST 域）len' = n−lb ≤ n 恒不溢出 → 0，
 *   UB 布局逐字节不变。host 守卫 CtpsvKernelUbFeasible 同式（LOWER 多 chunk 档边界
 *   缩移：incx=1 N 10528→10496 / T·C 9632→9536；UPPER 不变）。
 *   L1 杆负知识留档（issue_20260928_opt4-l1-sv-pair-scheduler-anchor_01）：V1 UNIT
 *   no-op 写回+S_V 对删除两变体实测 T2/T8 scalar +2~3% 反向劣化——S_V 对为 bisheng
 *   循环调度锚，本杆不触碰该同步点。
 *
 * 迭代四 opt4 L4（OPT4_PLAN L4；2026-09-28，位级杆）：V2 dot 级联融合单趟化——
 *   DotChunk 尾部两条独立标量级联（sR/sI 各自两级级联 + 各一组 V_S 对）合并为
 *   CtpsvCascadeSum2 单趟：一级 RRS 分写 w[0..rep) / w[64..64+rep)（repeat 对齐界），
 *   二级单趟 2 repeat → w2[0]=ΣsR / w2[1]=ΣsI；rep=1 走免二级最短路径（rep 分支防
 *   二级劣化，ctbsv CtbsvCascadeSlot 落槽形态直迁）。每步省 3 Duplicate + 6
 *   PipeBarrier + 1 RRS + 1 组 V_S 事件对。位级等价（部分和集合/落位与二级树窗口
 *   同构）；w/w2 域复用（w[0..128) ⊂ 2·W_SPAN），零 UB/守卫变更。仅 V2 模板
 *   （T3-T6/T9-T12）DotChunk 路径触及。
 *
 * 迭代五 opt5 杆 1（OPT5_PLAN 杆 1；2026-09-28，位级杆）：stpsv 单调免清零——
 *   SolveRowDot 入口一次预清零 tR/tI 全宽 + w[0..2·W_SPAN)，整列单块档
 *   （monoZero = chunkLen==n，host 策略 ⇔ FAST n≤4096 / GATHER·STRIDED n≤2048）
 *   删每步 3 Duplicate（DotChunk 尾垫 [lenA,lenR)×2 + CascadeSum2 二级读域
 *   w[0,128)）+ 1~2 PipeBarrier。单调性论证：LOWER V2 回代 len'=n-lb（lb 随 i 降
 *   非增）/ UPPER V2 前代 len=i（随 i 升非降）→ lenA/rep 逐步非降 → 尾垫与二级
 *   读域外 lane 从未被写 → Init 零恒存（stpsv CascadeMono 单调免清零同域先例）。
 *   多 chunk 档（chunkLen<n，rep 逐 chunk 组界回摆非单调）谓词 false 走每步清零
 *   遗留路径零风险。位级等价（写域/写值/消费域全同）；仅 V2 模板触及，V1（T1/T2/
 *   T7/T8）不实例化 DotChunk/SolveRowDot——控制序逐位保持。
 *
 * 迭代五 opt5 杆 2（OPT5_PLAN 杆 2；2026-09-28，非位级杆）：B-lite BLK2 —— V2
 *   LOWER T/C UNIT（T10/T12）两步并一：行对 (i,i−1) 单迭代 = 双列窗口（列 i−1/i
 *   行 [lb,n)，lb=i&~7）双 DMA 双独立 PF id（3/4）+ 顺序双列 MAC（DotWindow×2，
 *   同 b 切片、同掩蔽域 [lb,i] 乘后掩 t）+ UNIT 闭式标量尾项 A[i,i−1]·x_i 后置
 *   （stpsv/ctrsv BLK2 闭式缩链先例；probe_blk2_v2 TU 分离探针 228/228 全绿后
 *   上线）。非位级：尾项移出 RRS 树 → 求和序变化（CPU 同序模型交叉验证 + 全精度
 *   门 + Inf 域容差验证）。分发谓词 = FAST && chunkLen==n && dbuf（dbuf 承载
 *   2-slot 足迹守卫，host 零变更）；NON_UNIT/UPPER/多 chunk/超 UB 档走遗留
 *   SolveRowDot。T9/T11/T3-T6/V1 全部不实例化新方法（if constexpr 丢弃分支）。
 *
 * 迭代五 opt5 杆 3（OPT5_PLAN 杆 3 变体；2026-09-28，位级杆 vs 杆 2）：BLK2 融合
 *   级联 4 槽——两窗 level-1 RRS 部分和分写 w[0..rep)∪[64..64+rep)（窗1）与
 *   [128..128+rep)∪[192..192+rep)（窗2），二级单趟 4 repeat → w2[0..4)=
 *   [s1R,s1I,s2R,s2I] + 单 V_S 对收尾（CascadeSum4）——与两趟 CascadeSum2 逐
 *   组件求和树同构 → 位级一致（vs 杆 2 形态）。每对省 1 条二级 RRS + 1 组 V_S
 *   对 + 2 GetValue；零 UB 增量（w 4×128 槽 ⊂ 2·W_SPAN 原域）。
 *   杆 3b 实验否决留档：a 平面 [len,lenA) dummy 尾清零向量化（Duplicate t 尾）
 *   ——dst 子偏移 len=n−lb ≡ n (mod 8) 非 32B 对齐，真机 507035（探针首例即炸）；
 *   该窄域标量环为结构必需（兼防 a_dummy=Inf 经 ×b_pad 传播）。
 *   收益（官方口径相邻 A/B，杆 2 终态为基）：T10/T12 再 −6.4~−6.7%。
 *
 * opt5 r2 杆 4（2026-09-28，非位级杆）：V1 BLK2（B2LP）—— V1 LOWER N UNIT
 *   （T8）前代对 (j, j+1) 单迭代（probe_blk2_v1 TU 分离探针真机验证后上线：
 *   corr 114/114 + perf 全带 −14~16% + Inf 域留档）。双窗双 slot 双 DMA
 *   （窗口1 = 列 j 行 [lb, n)、窗口2 = 列 j+1 行 [lb2, n)，lb = j&~7 /
 *   lb2 = (j+2)&~7）；x_{j+1} = b_{j+1} − A[j+1,j]·x_j 标量闭式（尾项自窗1
 *   lane 标量读）；掩蔽域 [lb, j+2)/[lb2, j+2)（V1 a 平面前置清零）；每对
 *   V_S/S_V/PF wait/DrainV 各 2→1。单装载（两窗物理并域）不可行已证：两窗
 *   距 ≡ n−j−1 (mod 4) 随对递进交替，vred src 32B / Axpy dst 32B 双对齐无解。
 *   有限性守卫按对重推导（列 j / 列 j+1 独立守卫）+ 回退 Axpy src 偏移
 *   pref = r0−lb ∈ {0,8}（aR[k] ↔ row lb+k 正确行映射）。非位级（闭式二项
 *   vs 向量 lane 三项舍入序）→ CPU 同序模型交叉验证 + 全精度门。分发谓词 =
 *   FAST && chunkLen==n && dbuf；NON_UNIT（T7）/UPPER（T1/T2）零触碰。
 *   附带留档（Inf 域观察，非缺陷定谳）：遗留 SolveColumnAxpy merged 回退
 *   路径 TrailingAxpy(StepSegStart(j), len) 读 aR[0]（↔ row lb+k）而消费行
 *   为 r0+k——pref = r0−lb ∈ {0,8}（T8 UNIT 偶 j 恒 8）lane 错位理论存在；
 *   实证不可判别（复数 Inf soup 塌缩：晚位 (Inf,0) 判别注入下 A/B/golden
 *   三方全 diverge ~95%，门域外信息行；有限域回退为死分支）。BLK2 新路径
 *   取正确偏移语序。
 *
 * opt5 r2 杆 4b（2026-09-28，位级杆）：V2 BLK2（T10/T12）MAC 计数收紧——
 *   DotWindowL1 6 条向量计数 lenA→len（count 形式 set_mask_count+
 *   set_vector_mask(0,count) 精确元素界，dav_c220 impl 实证）+ ConsumeWindow
 *   dummy 尾 [len,lenA) 标量清零环删除（≤7×2 SetValue×2 窗/对，n≡2/1 (mod 8)
 *   档 24~28 次/对）。a 尾垃圾 lane 不再进入任何向量 MAC（0×Inf=NaN 暴露面
 *   一并封闭）；t 尾垫 [len,lenA) 恒为入口预清零零值（len 逐对非降，杆 1
 *   论证同域）→ RRS 消费域值逐位一致——位级等价（vs r2 杆 4 终态）。仅
 *   BLK2 路径触及（T10/T12 FAST 单 chunk dbuf 档）；遗留 DotChunk/
 *   ConsumeColChunk（T3-T6/T9/T11 + 多 chunk 档）零触碰。
 *
 * 迭代一穿刺结论整合（PROBE_SUMMARY 2026-09-26）：
 *   1) probe_deinterleave：AoS→SoA 解交织主路径 = vreducev2（66/66 bit-exact，2.3× 优于
 *      Gather），kernel 入口 __vector__ 属性、Duplicate 预清零 + align64 预留（x 装载
 *      路径保留）、rep=ceil(2·ceil8(n)/64)；incx≠1 = 物理跨度装载 + Gather 绝对字节
 *      偏移表（仿射 first+k·step，源基址 0），pad lane 显式清零。
 *   2) probe_complex_div：朴素复除 p·conj(d)/|d|² 定谳锁定（Smith 不启用）。
 *
 * incx 路径（运行期选择，不增编译分支，stpsv/ctrsv 交付范式）：
 *   FAST（incx=1）   整段 DataCopyPad + vreducev2 解交织双平面；
 *   GATHER（incx≠1） 物理跨度整段装载像（bufB 尾接）+ Gather 仿射偏移表映射紧凑平面；
 *                    写回 = 单源拼接张量 [bR|bI|装载像] 分片 Gather（gap 槽位自指原值保持）
 *                    + DataCopyPad 整段回写；
 *   STRIDED（护栏）  联合 UB 不等式超 184KB 或 span8 > 6144 时回退逐元素 strided 微拷贝
 *                    （blockCount=1/blockLen=8B，gap 物理零触碰，任意跨度功能正确）。
 *
 * Design Contract（DESIGN.md §8）硬约束：
 *  - 单 AIV 向量内核：KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)，串行依赖链
 *    单 AIV 核闭合（blockIdx==0 承接全部串行链）；核数 GetBlockNum/GetBlockIdx
 *    动态获取禁止写死。
 *  - 12 编译期模板一一对应枚举值域 {121,122}×{111,112,113}×{131,132}，运行期零分支；
 *    OP_C 独立分支禁止折叠为 OP_T（host/kernel 双侧）。
 *  - 右视 axpy（OP_N）：解出 x_j 后以 AP 第 j 列去对角段做 trailing 更新；
 *    左视 dot（OP_T/C）：x_i = (b_i − dot(AP 列 i 已解段, x 已解段))/d；
 *    对角位置：LOWER=列首 / UPPER=列末；UNIT 对角零读取。
 *  - TilingDataDevice GM 传递：reinterpret_cast<__gm__ CtpsvTilingDataDevice*> 解析。
 *  - API 白名单（DESIGN §3.6/3.7）：DataCopyPad / Axpy / Mul / Add / Sub / Muls / Adds /
 *    RepeatReduceSum 级联 / Gather / vreducev2 / SetMaskCount+SetVectorMask(COUNTER) 成对 /
 *    SetFlag-WaitFlag / Duplicate / LocalTensor::GetValue/SetValue（UB 级，黑名单仅 GM 侧）；
 *    禁逐元素标量 GM 访问；repeatTimes ≤ 255 分批（chunk ≤ 2048 → rep ≤ 32）；
 *    WholeReduceSum 禁用于 count>64 全和（Mul+RRS 级联承载）。
 *
 * 真机实证纪律（ctrsv/stpsv arch22 同芯片先例，逐位镜像）：
 *  - raw 向量指令（RRS）前显式 SetMaskNorm + SetVectorMask 全开位掩码兜底（count 模式
 *    残留掩码静默截断 raw 指令，ctrsv/probe 实证）；
 *  - counter mask 谓词为有状态硬件计数器：SetMaskCount 计数区内只包纯 Axpy，
 *    SetVectorMask 显式成对 + 即时 SetMaskNorm/ResetMask 收口；Gather/vreducev2 一律在
 *    normal mask 区执行（ctrsv 507035 崩溃实证）；
 *  - DataCopyPad dummy 填充纪律（ctrsv 迭代二实证）：blockLen 非 32B 整块时 dst 尾部被
 *    dummy 填充 = 首元素值——vred 提取出非零 pad 系数，须 vred 后显式 SetValue 清零
 *    [len, lenA) pad lane（消费域三重护栏见上「迭代三 2)」）；
 *  - incx≠1 双平面 pad [n, nPad) 显式清零（ctrsv issue_20260827_tunit_incx_nan_01：
 *    残留 Inf 经 a_pad(0)·Inf=NaN 污染 dot 归约）；
 *  - col slot MTE2 覆写前 V_MTE2 排空对（stpsv DrainVBeforeColLoad -O2 竞态修复形态；
 *    迭代三流水化后由 IssueColChunk 内 Drain 承载，头段移驻 bufDiag 后不再需要）；
 *  - 事件 id 写死顺序复用（ctrsv issue_20260826_dbuf_multilaunch_hang_01 规避形态）；
 *    每对 HardEvent 独立 id 空间，全局唯一编号；预取装载独立 id（MTE2_V_PF=3）与
 *    微拷贝 id（MTE2_V=2）严格各自交替 set/wait；
 *  - CANN 9.1.0 UB→GM DataCopyPad 仅 3 参形态（stpsv/ctrsv 实证）。
 *
 * UB 规划（DESIGN §3.9/3.10；192KB 上限，向量可用估算 184KB）：
 *   bufB      2·nPad + (GATHER: 2·span8) float      bR|bI 双平面常驻 + 跨度装载像（gap 保持源）
 *   bufXTrans FAST: align64(2·ceil8(n)) / GATHER: 2·WB_CHUNK float
 *              （装载 vreducev2 源 / 写回交织·分片暂存；STRIDED 不分配）
 *   bufCol    (dbuf ? 2 : 1) · align64(2·chunkPad+kColRideFloats) float
 *              AP 列段 AoS 交错装载双 slot 轮转（look-ahead 预取乒乓；超 UB 预算档
 *              自动退单 slot 串行，足迹与迭代二一致；opt2 P0 kColRideFloats = 搭车模板
 *              +64 float（前缀 + vred src 偏移越读护栏）/ 无搭车模板 0（原布局））
 *   bufAR/AI  align64(chunkPad) float 各            列段解交织实/虚平面（vred 目的）
 *   bufTR/TI/T2 align64(chunkPad) float 各（仅 V2） 复 dot 逐元素部分和 + 归约 scratch
 *   bufW      2·W_SPAN float（仅 V2）               RRS 级联 w/w2 部分和缓冲
 *   bufOffElem/bufOffOdd xLen8 int32 各（GATHER）   逻辑 i→装载像字节偏移表（实/虚伴随）
 *   bufOffWb  2·WB_CHUNK int32（GATHER）            写回分片逐 float 偏移表
 *   bufEl     8 float（STRIDED）                    单元素微拷贝
 *   bufDiag   16 float                              对角元 8B 微拷贝 + 头段微拷贝（迭代三移驻；
 *              opt2 P0 后仅遗留路径使用：T1 对角 / 主段空尾步）
 *   bufOff    256 int32                             FAST 写回交织表（Init 一次复用）
 *   最重口径（n=2048/|incx|=3/V2/GATHER）dbuf 关（+16KB 超预算）≈ 156KB ≤ 184KB；
 *   n=4096/incx=1/V2 dbuf 开 ≈ 139KB；n=4096/incx=1/V2 dbuf 关 ≈ 123KB（迭代二口径）。
 */

#ifndef ACLBLAS_CTPSV_KERNEL_H
#define ACLBLAS_CTPSV_KERNEL_H

#include <cstdint>
#include "kernel_operator.h"
#include "cann_ops_blas_common.h"
#include "ctpsv_tiling_data.h"
#include "ctpsv_kernel_utils.h"

using namespace AscendC;

constexpr uint32_t CTPSV_FLOAT_ALIGN = 8;    // VEC 指令计数/子偏移/缓冲尺寸 8 float（32B）对齐
constexpr uint32_t CTPSV_GATHER_CHUNK = 256; // FAST 写回交织 Gather 分片（float 计，ctrsv 先例 ≤ 256）
constexpr uint32_t CTPSV_WB_CHUNK = 1024;    // GATHER 写回分片（复数计，8 倍数）
constexpr uint32_t CTPSV_W_SPAN = 256; // RRS 级联 w/w2 部分和缓冲宽度（单级 rep ≤ 255 → 域 ≤ 256）
constexpr int32_t CTPSV_SPAN8_MAX = 6144; // GATHER 物理跨度装载像上限（2·span8 float ≤ 48KB，
                                          //   ctrsv CTRSV_SPAN8_MAX 同款；超界走 STRIDED）
                                          // 小 n 标量快路径门槛 CTPSV_SMALL_N_MAX 见
                                          // ctpsv_tiling_data.h（host/kernel 共享单一真源）
constexpr int64_t CTPSV_UB_BUDGET_BYTES = 184LL * 1024LL; // §3.10 向量可用 UB 估算（2201 192KB/核）

// 事件 id（写死顺序复用，全局唯一编号；每对 HardEvent 独立 id 空间）
constexpr uint32_t CTPSV_EVT_Z0 = 0;         // V_MTE2：清零/排空→装载
constexpr uint32_t CTPSV_EVT_MTE2_V = 2;     // MTE2_V：微拷贝（x 段/对角/头段）→计算
constexpr uint32_t CTPSV_EVT_MTE2_V_PF = 3;  // MTE2_V：AP 列段预取装载→vred 消费（迭代三独立 id，
                                             //   与微拷贝 MTE2_V(2) 交替 set/wait 不互扰）
constexpr uint32_t CTPSV_EVT_MTE2_V_PF2 = 4; // opt5 杆 2：BLK2 双窗预取第二独立 id（窗2 装载
                                             //   →vred 消费；与 PF(3) 各自单飞 set→wait 不重入）
constexpr uint32_t CTPSV_EVT_V_S = 5;        // V_S：向量写→标量读（级联收尾 / axpy 步首 b_j 读）
constexpr uint32_t CTPSV_EVT_S_V = 6;        // S_V：标量写（x_i SetValue）→向量读
constexpr uint32_t CTPSV_EVT_V_MTE3 = 7;     // V→MTE3：解平面写→写出
constexpr uint32_t CTPSV_EVT_MTE3_V = 8;     // MTE3→V：写出完成→缓冲复用（分片循环）
constexpr uint32_t CTPSV_EVT_S_MTE3 = 9;     // S→MTE3：标量写→写出

// 上取整对齐（VEC 计数 8 float / vreducev2 缓冲 64 float 纪律）
__aicore__ inline int32_t CtpsvCeil(int32_t x, int32_t align) { return (x + align - 1) & ~(align - 1); }
__aicore__ inline int32_t CtpsvCeil8(int32_t x) { return CtpsvCeil(x, CTPSV_FLOAT_ALIGN); }
__aicore__ inline int32_t CtpsvCeil64(int32_t x) { return CtpsvCeil(x, 64); }
__aicore__ inline int32_t CtpsvMin(int32_t a, int32_t b) { return (a < b) ? a : b; }

enum class CtpsvUplo { UPPER, LOWER };
enum class CtpsvTrans { OP_N, OP_T, OP_C };
enum class CtpsvDiag { UNIT, NON_UNIT };
enum class CtpsvXPath { FAST, GATHER, STRIDED }; // incx=1 连续 / 跨度+Gather / 护栏回退

// raw 向量指令前置兜底：恢复 normal 全开位掩码（ctrsv 真机纪律：count 模式残留掩码会静默截断 raw 指令）
__aicore__ inline void CtpsvMaskFull()
{
    SetMaskNorm();
    SetVectorMask<float>((uint64_t)-1, (uint64_t)-1);
}

// vreducev2 解交织封装（step=1 实平面 / step=2 虚平面；ctrsv LoadX/LoadColumnChunk
// 同参数形态，AoS 交错源 → SoA 平面目的）
__aicore__ inline void CtpsvVred(
    const LocalTensor<float>& dst, const LocalTensor<float>& src, uint32_t rep, uint32_t step)
{
    vreducev2(
        reinterpret_cast<__ubuf__ uint32_t*>(dst.GetPhyAddr()), reinterpret_cast<__ubuf__ uint32_t*>(src.GetPhyAddr()),
        nullptr, rep, 1, step, 8, 8);
}

// 仿射字节偏移表构造（ctrsv CtrsvBuildAffineOffsets 交付先例：SetValue 头 8 项 + Adds 块平移；
// t[k] = first + k·step，k ∈ [0, len8)；len8 为 8 的倍数；step 可为负（xNeg 递减表））
__aicore__ inline void CtpsvBuildAffineOffsets(LocalTensor<int32_t>& t, int32_t first, int32_t step, int32_t len8)
{
    for (int32_t k = 0; k < static_cast<int32_t>(CTPSV_FLOAT_ALIGN); k++) {
        t.SetValue(k, first + k * step);
    }
    PipeBarrier<PIPE_V>();
    for (int32_t g = static_cast<int32_t>(CTPSV_FLOAT_ALIGN); g < len8; g += static_cast<int32_t>(CTPSV_FLOAT_ALIGN)) {
        Adds(t[g], t, g * step, static_cast<int32_t>(CTPSV_FLOAT_ALIGN));
        PipeBarrier<PIPE_V>();
    }
}

/**
 * opt4 L4：V2 复 dot 级联融合单趟化（sR/sI 双平面一次收束，ctbsv CtbsvCascadeSlot
 * 落槽形态直迁）。对两条独立标量级联调用：省 3 Duplicate + 6 PipeBarrier +
 * 1 RRS + 1 组 V_S 事件对/步（rep 分支防二级劣化前科——rep=1 走免二级最短路径）。
 *   rep = 1：双平面各 1 条 RRS 直落 w[0] / w2[0]（源 [lenA,64) 尾垫已零），免
 *           Duplicate 与二级（读域恰为已垫零的 tR/tI[0..64)）；
 *   1 < rep ≤ 64（调用域不变式：lenA = ceil8(len') ≤ chunkLen+8 ≤ 4104 → rep ≤ 64；
 *           FAST n≤4096 档 len' ≤ n ≤ 4096 → rep ≤ 64，GATHER/n>4096 档更小）：
 *           一级分写 w[0..rep) / w[64..64+rep)（repeat 对齐界 64，子偏移 64 float
 *           = 256B 对齐），二级单趟 2 repeat → w2[0]=ΣsR / w2[1]=ΣsI。
 *   位级等价（vs 两条独立级联）：一级部分和集合与落位值不变；二级各 repeat
 *   树求和窗口（64 lane，值相对位置不变）与旧单 repeat 窗口同构 → fp 结果逐位一致。
 * opt5 杆 1（stpsv 单调免清零）：wPad=true（整列单块档 chunkLen==n，V2 步长单调：
 * LOWER len'=n-lb 随 i 降非降 / UPPER len=i 随 i 升非降 → rep 逐步非降）时，二级读域
 *   [rep,64)/[64+rep,128) 从未被更大 rep 的一级写过 → SolveRowDot 入口一次预清零
 *   w[0..2·W_SPAN) 即恒为零垫，删每步 Duplicate(w,0,128)+PipeBarrier。位级等价：
 *   读域值（0.0f）与写域（本步一级 [0,rep)∪[64,64+rep)）均不变。多 chunk 档
 *   （chunkLen<n）rep 逐 chunk 不单调，wPad=false 走原每步清零零风险。
 */
__aicore__ inline void CtpsvCascadeSum2(
    const LocalTensor<float>& tR, const LocalTensor<float>& tI, const LocalTensor<float>& w,
    const LocalTensor<float>& w2, int32_t len, float& sR, float& sI, bool wPad)
{
    const int32_t rep = (len + 63) >> 6;
    if (rep <= 0) {
        sR = 0.0f;
        sI = 0.0f;
        return;
    }
    if (rep == 1) {
        CtpsvMaskFull();
        RepeatReduceSum<float, false>(w, tR, 1, 0, 0, 1, 1, 8);  // w[0] = Σ tR
        RepeatReduceSum<float, false>(w2, tI, 1, 0, 0, 1, 1, 8); // w2[0] = Σ tI
        SetFlag<HardEvent::V_S>(CTPSV_EVT_V_S);                  // V 写部分和 → S 标量读
        WaitFlag<HardEvent::V_S>(CTPSV_EVT_V_S);
        sR = w.GetValue(0);
        sI = w2.GetValue(0);
        return;
    }
    if (!wPad) {
        Duplicate(w, 0.0f, 128); // 二级读域 [0,128) 预清零（含 [rep,64)/[64+rep,128) 零垫）
        PipeBarrier<PIPE_V>();
    }
    CtpsvMaskFull();
    RepeatReduceSum<float, false>(w, tR, static_cast<uint8_t>(rep), 0, 0, 1, 1, 8);     // sR 部分和
    RepeatReduceSum<float, false>(w[64], tI, static_cast<uint8_t>(rep), 0, 0, 1, 1, 8); // sI 部分和
    PipeBarrier<PIPE_V>();
    CtpsvMaskFull();
    RepeatReduceSum<float, false>(w2, w, 2, 0, 0, 1, 1, 8); // 二级单趟：w2[0]=ΣsR / w2[1]=ΣsI
    SetFlag<HardEvent::V_S>(CTPSV_EVT_V_S);                 // V 写部分和 → S 标量读
    WaitFlag<HardEvent::V_S>(CTPSV_EVT_V_S);
    sR = w2.GetValue(0);
    sI = w2.GetValue(1);
}

// ==========================================================================
//  Kernel class — parameterized on uplo / trans / diag（12 编译期模板）
//    TRANS=OP_N   → 右视列 axpy（V1，T1/T2/T7/T8）
//    TRANS=OP_T/C → 左视列 dot（V2，T3-T6/T9-T12；OP_C 共轭三处承载）
//    incx 路径为运行期参数选择（FAST/GATHER/STRIDED），不增编译分支
// ==========================================================================

template <CtpsvUplo UPLO, CtpsvTrans TRANS, CtpsvDiag DIAG>
class CtpsvKernel {
public:
    __aicore__ inline CtpsvKernel() {}

    __aicore__ inline void Init(GM_ADDR apGm, GM_ADDR xGm, GM_ADDR tilingGm)
    {
        blockIdx = GetBlockIdx();
        blockNum = GetBlockNum(); // 动态获取核信息，禁止写死（Design Contract Programming Model）

        const __gm__ CtpsvTilingDataDevice* t = reinterpret_cast<__gm__ CtpsvTilingDataDevice*>(tilingGm);
        n = static_cast<int32_t>(t->n);
        incx = t->incx;
        incxAbs = (incx < 0) ? -incx : incx;
        xNeg = (incx < 0) ? 1 : 0;
        chunkLen = static_cast<int32_t>(t->chunk);
        if (chunkLen <= 0 || chunkLen > n) {
            chunkLen = n; // 钳位（n=1 等退化档；chunk ∈ {n, 2048, 1024} → 分片步进保 8 对齐）
        }

        nPad = CtpsvCeil64(n); // bR/bI 平面基址（虚平面 vreducev2 目的起址 32B 对齐）
        iPlane = nPad;
        chunkPad = CtpsvCeil8(chunkLen);
        xLen8 = CtpsvCeil8(n);
        // opt4 L3：LOWER 头段窗口化 plane 增量（运行期条件）。chunk0 merged 消费长
        //   len' = ml + 8 仅在多 chunk 档（chunkLen < n：GATHER n>2048 / FAST n>4096）
        //   可达 chunkPad+8 → aR/aI（V1）与 tR/tI/t2（V2）需 +64 对齐界（ceil64(lenA') ≤
        //   align64(chunkPad)+64）；整列单块档（chunkLen == n，含全部 n≤4096 FAST 域）
        //   len' = n-lb ≤ n 恒不溢出 → ride=0，UB 布局与迭代四 opt3 终态逐字节一致。
        //   host 守卫 CtpsvKernelUbFeasible 同式（issue_20260926 前科纪律）。
        planeRide = ((UPLO == CtpsvUplo::LOWER) && (chunkLen < n)) ? 64 : 0;
        // x 物理跨度（int64 中间积防上溢：incx 域 ±2147483647；n=0 不进 kernel）
        span64 = (n <= 1) ? 1 : 1 + static_cast<int64_t>(n - 1) * incxAbs;

        // GM：AP 逻辑长度 n(n+1)/2 复数 = 2·n(n+1)/2 float（uint64 中间积后截 uint32，
        // n ≤ 65535 域内 < 2^32）；x 物理跨度 span 复数 = 2·span float
        const uint64_t apCount = static_cast<uint64_t>(n) * (static_cast<uint64_t>(n) + 1) / 2;
        apGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(apGm), static_cast<uint32_t>(2ULL * apCount));
        xGM.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(xGm), static_cast<uint32_t>(2ULL * static_cast<uint64_t>(span64)));

        // ── incx 路径选择（stpsv/probe_neg_incx 同款护栏：联合 UB 不等式 + span8 上限）──
        //   GATHER 需求 = [双平面 + 跨度装载像](2nPad+2span8) + 偏移表 2·xLen8 + 写回表/暂存
        //                2·WB_CHUNK×2 + 求解器缓冲；超 184KB 或 span8 > 6144 → STRIDED 回退
        //                （逐元素微拷贝，任意跨度功能正确，gap 物理零触碰）。
        const int64_t span8 = (span64 + 7) / 8 * 8;
        int64_t solverFloats = static_cast<int64_t>(CtpsvCeil64(2 * chunkPad + kColRideFloats)) +
                               2LL * static_cast<int64_t>(CtpsvCeil64(chunkPad) + planeRide) + 16LL + 256LL;
        // （col 项 = kColRideFloats：opt2 P0 搭车前缀 + vred src 偏移护栏，与 colFloats 同式；
        //   plane 项 +planeRide = opt4 L3 头段窗口化多 chunk 档增量，host 守卫同式）
        if constexpr (TRANS != CtpsvTrans::OP_N) {
            solverFloats += 3LL * static_cast<int64_t>(CtpsvCeil64(chunkPad) + planeRide) +
                            2LL * static_cast<int64_t>(CTPSV_W_SPAN); // tR/tI/t2 + w/w2（仅 dot）
        }
        if (incx == 1) {
            path = CtpsvXPath::FAST;
        } else {
            const int64_t ubNeedGather =
                (2LL * nPad + 2LL * span8 + 2LL * xLen8 + 2LL * CTPSV_WB_CHUNK + 2LL * CTPSV_WB_CHUNK + solverFloats) *
                    4 +
                2048;
            if (ubNeedGather <= CTPSV_UB_BUDGET_BYTES && span8 <= static_cast<int64_t>(CTPSV_SPAN8_MAX)) {
                path = CtpsvXPath::GATHER;
                spanI = static_cast<int32_t>(span64); // 护栏内必有界（span8 ≤ 6144）
                span8I = static_cast<int32_t>(span8); // 装载像对齐宽度（InitBuffer 尺寸纪律）
            } else {
                path = CtpsvXPath::STRIDED;
            }
        }

        InitUBuffers();
    }

    // ── UB 规划（见文件头；GATHER 装载像接在 bufB 平面后 = 写回单源拼接张量）──
    __aicore__ inline void InitUBuffers()
    {
        const uint32_t bFloats =
            2U * static_cast<uint32_t>(nPad) + ((path == CtpsvXPath::GATHER) ? 2U * static_cast<uint32_t>(span8I) : 0U);
        pipe.InitBuffer(bufB, bFloats * sizeof(float));
        if (path != CtpsvXPath::STRIDED) {
            const uint32_t xtFloats =
                (path == CtpsvXPath::FAST) ? static_cast<uint32_t>(CtpsvCeil64(2 * xLen8)) : 2U * CTPSV_WB_CHUNK;
            pipe.InitBuffer(bufXTrans, xtFloats * sizeof(float));
        }
        if (path == CtpsvXPath::GATHER) {
            pipe.InitBuffer(bufOffElem, static_cast<uint32_t>(xLen8) * sizeof(int32_t));
            pipe.InitBuffer(bufOffOdd, static_cast<uint32_t>(xLen8) * sizeof(int32_t));
            pipe.InitBuffer(bufOffWb, 2U * CTPSV_WB_CHUNK * sizeof(int32_t));
        }
        if (path == CtpsvXPath::STRIDED) {
            pipe.InitBuffer(bufEl, 8U * sizeof(float));
        }
        // ── 迭代三：col 双 slot（look-ahead 预取乒乓）UB 决策 ──
        //   联合足迹（含双 col slot）≤ 184KB 时启用 dbuf；超预算档自动退回单 slot 串行
        //   （单 slot 足迹与 host 守卫式一致——host 侧 CtpsvKernelUbFeasible
        //   按最小足迹守卫，kernel 永不超预算分配，无需 host 变更）。
        colFloats = static_cast<int32_t>(CtpsvCeil64(2 * chunkPad + kColRideFloats));
        // kColRideFloats：opt2 P0 搭车 col slot 增量——LOWER（前缀+vred src 偏移越读护栏）
        //   与 UPPER V2 NON_UNIT（末块对角+1 元素护栏）= 64 float；T1/T2/T4/T6（无搭车）
        //   = 0 → UB 布局与迭代三逐字节一致（T2 档实测 col 布局漂移 +84µs scalar 税，
        //   bisheng 二进制恒等对照已排除分支折叠因素，故无搭车档回退原布局）。
        //   slot 起址保持 64-float/256B 对齐（+64 不破 64-float 对齐界）。
        int64_t footFloats = static_cast<int64_t>(bFloats) + static_cast<int64_t>(colFloats) +
                             2LL * static_cast<int64_t>(CtpsvCeil64(chunkPad) + planeRide) + 16LL + 256LL;
        if (path == CtpsvXPath::FAST) {
            footFloats += static_cast<int64_t>(CtpsvCeil64(2 * xLen8));
        } else if (path == CtpsvXPath::GATHER) {
            footFloats += 2LL * CTPSV_WB_CHUNK + 2LL * xLen8 + 2LL * CTPSV_WB_CHUNK; // xtrans + off表 + offWb
        } else {
            footFloats += 8LL;
        }
        if constexpr (TRANS != CtpsvTrans::OP_N) {
            footFloats += 3LL * static_cast<int64_t>(CtpsvCeil64(chunkPad) + planeRide) +
                          2LL * static_cast<int64_t>(CTPSV_W_SPAN);
        }
        dbuf = (footFloats + static_cast<int64_t>(colFloats)) * 4 + 2048 <= CTPSV_UB_BUDGET_BYTES;
        pipe.InitBuffer(bufCol, static_cast<uint32_t>(colFloats * ((dbuf) ? 2 : 1)) * sizeof(float));
        pipe.InitBuffer(bufAR, static_cast<uint32_t>(CtpsvCeil64(chunkPad) + planeRide) * sizeof(float));
        pipe.InitBuffer(bufAI, static_cast<uint32_t>(CtpsvCeil64(chunkPad) + planeRide) * sizeof(float));
        if constexpr (TRANS != CtpsvTrans::OP_N) {
            pipe.InitBuffer(bufTR, static_cast<uint32_t>(CtpsvCeil64(chunkPad) + planeRide) * sizeof(float));
            pipe.InitBuffer(bufTI, static_cast<uint32_t>(CtpsvCeil64(chunkPad) + planeRide) * sizeof(float));
            pipe.InitBuffer(bufT2, static_cast<uint32_t>(CtpsvCeil64(chunkPad) + planeRide) * sizeof(float));
            pipe.InitBuffer(bufW, 2U * CTPSV_W_SPAN * sizeof(float));
        }
        pipe.InitBuffer(bufDiag, 16U * sizeof(float));
        pipe.InitBuffer(bufOff, CTPSV_GATHER_CHUNK * sizeof(int32_t));

        BuildInterleaveOffsets(); // FAST 写回交织偏移表（Init 一次永久复用）
    }

    __aicore__ inline void Process()
    {
        if (blockIdx != 0) {
            return; // 串行依赖链单 AIV 核闭合（首版 useCoreNum=1，多核冗余分发保险）
        }
        LoadX();
        if constexpr (TRANS == CtpsvTrans::OP_N) {
            // opt5 r2 杆 4：V1 BLK2 分发（仅 LOWER/N/UNIT = T8 实例化；谓词 =
            //   FAST 整列单块 + dbuf 双 slot——非谓词档（GATHER/STRIDED、多
            //   chunk n>4096、超 UB 退单 slot）走遗留 SolveColumnAxpy 零风险；
            //   dbuf 谓词承载 2-slot 足迹守卫，host 侧零变更）
            if constexpr (UPLO == CtpsvUplo::LOWER && DIAG == CtpsvDiag::UNIT) {
                if (path == CtpsvXPath::FAST && chunkLen >= n && dbuf) {
                    SolveColumnBlk2();
                } else {
                    SolveColumnAxpy();
                }
            } else {
                SolveColumnAxpy();
            }
        } else {
            // opt5 杆 2：B-lite BLK2 分发（仅 LOWER/T·C/UNIT 实例化；谓词 = FAST 整列
            //   单块 + dbuf 双 slot 可用——非谓词档（GATHER/STRIDED、多 chunk、UB 超
            //   预算退单 slot）走遗留 SolveRowDot 零风险；dbuf 谓词承载 2-slot 足迹
            //   守卫，host 侧零变更）。
            if constexpr (UPLO == CtpsvUplo::LOWER && DIAG == CtpsvDiag::UNIT) {
                if (path == CtpsvXPath::FAST && chunkLen >= n && dbuf) {
                    SolveRowDotBlk2();
                } else {
                    SolveRowDot();
                }
            } else {
                SolveRowDot();
            }
        }
        WriteBackX();
    }

private:
    // packed 列内元素偏移（复数元素单位）：UPPER 列 col 存行 [0,col] → colStart+row；
    // LOWER 列 col 存行 [col,n) → colStart+(row-col)（列内连续稠密段，MTE2 整段装载）
    __aicore__ inline int64_t ColElemOff(int32_t col, int32_t row) const
    {
        if constexpr (UPLO == CtpsvUplo::LOWER) {
            return CtpsvLowerColStart(col, n) + (row - col);
        } else {
            return CtpsvUpperColStart(col) + row;
        }
    }

    // 逻辑 i → 物理复数下标（Netlib ctpsv 语义：incx<0 反向遍历，逻辑 i ↔ 物理 (n-1-i)·|incx|）
    __aicore__ inline int64_t Phys(int32_t i) const
    {
        return (xNeg != 0) ? static_cast<int64_t>(n - 1 - i) * incxAbs : static_cast<int64_t>(i) * incxAbs;
    }

    // ── FAST 写回交织偏移表（Init 一次永久复用；ctrsv BuildInterleaveOffsets 先例）──
    //    offInter[2m] = m·4（bR[m] 字节）、offInter[2m+1] = (nPad+m)·4（bI[m] 字节），
    //    m < 128（GATHER_CHUNK/2）──
    __aicore__ inline void BuildInterleaveOffsets()
    {
        LocalTensor<int32_t> off = bufOff.Get<int32_t>();
        for (int32_t m = 0; m < static_cast<int32_t>(CTPSV_GATHER_CHUNK) / 2; m++) {
            off.SetValue(2 * m, m * static_cast<int32_t>(sizeof(float)));
            off.SetValue(2 * m + 1, (nPad + m) * static_cast<int32_t>(sizeof(float)));
        }
        PipeBarrier<PIPE_ALL>();
    }

    // ── Stage 0：x/b 装载 GM→UB 双平面常驻（逻辑紧凑序；三路径分解）──
    //  FAST：整段 DataCopyPad 交错装载 → vreducev2 解交织 bR/bI（迭代一交付形态）；
    //        暂存清零 pad（vred 尾 repeat 越界读不污染输出）；
    //  GATHER：物理跨度整段装载像（写回期 gap 保持源，bufB 尾接）→ Gather 仿射偏移表
    //        映射紧凑双平面（xNeg=1 递减仿射 first=8s(n-1) step=-8s，probe_deinterleave 定稿）；
    //        pad [n, nPad) 显式清零（残留 Inf 经 0·Inf=NaN 污染 dot 归约，ctrsv 实证陷阱）；
    //  STRIDED：护栏回退，逐元素 strided 微拷贝（blockCount=1/blockLen=8B，任意跨度）。
    __aicore__ inline void LoadXFast(const LocalTensor<float>& bAll)
    {
        LocalTensor<float> xIl = bufXTrans.Get<float>();
        const int32_t xIlFloats = CtpsvCeil64(2 * xLen8);
        Duplicate(xIl, 0.0f, static_cast<uint32_t>(xIlFloats)); // 清零 pad（vred 源纪律）
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE2>(CTPSV_EVT_Z0); // V 清零完成 → MTE2 方可写入
        WaitFlag<HardEvent::V_MTE2>(CTPSV_EVT_Z0);
        DataCopyExtParams cp{1, 8U * static_cast<uint32_t>(n), 0, 0, 0};
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        DataCopyPad(xIl, xGM, cp, pad);
        SetFlag<HardEvent::MTE2_V>(CTPSV_EVT_MTE2_V);
        WaitFlag<HardEvent::MTE2_V>(CTPSV_EVT_MTE2_V);
        const uint32_t rep = static_cast<uint32_t>((2 * xLen8 + 63) / 64);
        CtpsvVred(bAll, xIl, rep, 1);         // step=1 → 实平面 bR
        CtpsvVred(bAll[iPlane], xIl, rep, 2); // step=2 → 虚平面 bI
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void LoadXGather(LocalTensor<float> bAll)
    {
        LocalTensor<float> image = bAll[2 * nPad]; // 跨度装载像（写回期 gap 保持源）
        DataCopyExtParams cp{1, 8U * static_cast<uint32_t>(spanI), 0, 0, 0};
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        DataCopyPad(image, xGM, cp, pad); // 跨度整段装载
        SetFlag<HardEvent::MTE2_V>(CTPSV_EVT_MTE2_V);
        WaitFlag<HardEvent::MTE2_V>(CTPSV_EVT_MTE2_V);
        // 逻辑 i → 装载像字节偏移：xNeg=0 递增 i·8s / xNeg=1 递减 (n-1-i)·8s（仿射表）
        LocalTensor<int32_t> offElem = bufOffElem.Get<int32_t>();
        LocalTensor<int32_t> offOdd = bufOffOdd.Get<int32_t>();
        const int32_t s8 = incxAbs * 2 * static_cast<int32_t>(sizeof(float));
        if (xNeg == 0) {
            CtpsvBuildAffineOffsets(offElem, 0, s8, xLen8);
        } else {
            CtpsvBuildAffineOffsets(offElem, (n - 1) * s8, -s8, xLen8);
        }
        Adds(offOdd, offElem, static_cast<int32_t>(sizeof(float)),
             xLen8);             // 虚部 = 实部偏移 + 4B
        PipeBarrier<PIPE_ALL>(); // MTE2→V + S→V
        CtpsvMaskFull();         // Gather 须在 normal mask 区执行（ctrsv 507035 实证）
        Gather(bAll, image, offElem.ReinterpretCast<uint32_t>(), 0, n);
        Gather(bAll[iPlane], image, offOdd.ReinterpretCast<uint32_t>(), 0, n);
        for (int32_t k = n; k < nPad; k++) { // pad 缺口显式清零（NaN 陷阱防护，见上）
            bAll.SetValue(k, 0.0f);
            bAll.SetValue(iPlane + k, 0.0f);
        }
        PipeBarrier<PIPE_V>();
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void LoadXStrided(LocalTensor<float> bAll)
    {
        LocalTensor<float> el = bufEl.Get<float>();
        DataCopyExtParams cp{1, 8U, 0, 0, 0}; // blockCount=1 / blockLen=8B（单复数元素）
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        for (int32_t i = 0; i < n; ++i) {
            DataCopyPad(el, xGM[2 * Phys(i)], cp, pad);
            SetFlag<HardEvent::MTE2_V>(CTPSV_EVT_MTE2_V);
            WaitFlag<HardEvent::MTE2_V>(CTPSV_EVT_MTE2_V);
            bAll.SetValue(i, el.GetValue(0));
            bAll.SetValue(iPlane + i, el.GetValue(1));
        }
        for (int32_t k = n; k < nPad; k++) { // pad 缺口显式清零（同 GATHER 纪律）
            bAll.SetValue(k, 0.0f);
            bAll.SetValue(iPlane + k, 0.0f);
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void LoadX()
    {
        LocalTensor<float> bAll = bufB.Get<float>();
        if (path == CtpsvXPath::FAST) {
            LoadXFast(bAll);
        } else if (path == CtpsvXPath::GATHER) {
            LoadXGather(bAll);
        } else {
            LoadXStrided(bAll);
        }
    }

    // col slot MTE2 覆写前排空 V 单元读（stpsv DrainVBeforeColLoad -O2 竞态修复形态；
    //  迭代三：由 IssueColChunk 承载——目标 slot 上上一轮 vred 读完成后方可覆写）
    __aicore__ inline void DrainVBeforeColLoad()
    {
        SetFlag<HardEvent::V_MTE2>(CTPSV_EVT_Z0);
        WaitFlag<HardEvent::V_MTE2>(CTPSV_EVT_Z0); // V（此前 vred/Axpy 读）排空 → MTE2 方可覆写
    }

    // ══════════════════════════════════════════════════════════════════════
    //  迭代三：AP 列段 chunk 流水（look-ahead 预取，DESIGN §4.3「双缓冲深度/look-ahead」）
    //  全局 chunk 序列 = 12 模板各自的（步骤 × 步内 chunk）扁平序；每次 Consume 后发射
    //  全局后继（同步下一 chunk，或跨步 look-ahead 到后续首个非空步骤的首 chunk），
    //  MTE2 装载与 V 单元计算重叠。col 双 slot 轮转（dbuf）；串行档（dbuf=false）同
    //  Issue/Consume 结构、slot 恒 0（足迹与迭代二一致）。
    // ══════════════════════════════════════════════════════════════════════

    // 步骤主段起点/终点（V1 trailing / V2 dot 区共用式；头段 [s0, segStart) 标量承载）
    __aicore__ inline int32_t StepSegStart(int32_t s) const
    {
        if constexpr (UPLO == CtpsvUplo::UPPER) {
            return 0; // 前缀 [0, s)（起点 0 天然对齐）
        } else {
            const int32_t s0 = s + 1;
            return (CtpsvCeil8(s0) < n) ? CtpsvCeil8(s0) : n; // 头段后主段（8 对齐边界）
        }
    }
    __aicore__ inline int32_t StepSegEnd(int32_t s) const
    {
        if constexpr (UPLO == CtpsvUplo::UPPER) {
            return s;
        } else {
            return n;
        }
    }
    __aicore__ inline int32_t StepChunkCount(int32_t s) const
    {
        const int32_t st = StepSegStart(s);
        const int32_t en = StepSegEnd(s);
        return (en > st) ? (en - st + chunkLen - 1) / chunkLen : 0;
    }

    // ── opt2 P0：搭车装载几何（头段/对角并入列段 bulk 装载，DESIGN §3.5 变更版）──
    // chunk 装载区间一次算清（主段基/长各求值一次——IssueSuccessor 热路径禁 helper 链
    // 重复求值：实测 T2 档 StepSegStart×3 重算 = +50 cycles/步 ≈ +4% 墙钟，msprof
    // 定谳 +84µs 全在 scalar pipe）。返回装载基（LOWER chunk0 = 8 对齐回退 L）与装载长
    // （主段长 + 前缀 + UPPER V2 末块对角搭车 +1）。
    __aicore__ inline void ChunkLoad(int32_t s, int32_t c, int32_t& loadBase, int32_t& loadLen) const
    {
        const int32_t mb = StepSegStart(s) + c * chunkLen;         // 主段基（1 次）
        const int32_t ml = CtpsvMin(chunkLen, StepSegEnd(s) - mb); // 主段长（1 次）
        int32_t lb = mb;
        if (c == 0) {
            if constexpr (UPLO == CtpsvUplo::LOWER) {
                if constexpr (DIAG == CtpsvDiag::NON_UNIT) {
                    lb = s & ~(static_cast<int32_t>(CTPSV_FLOAT_ALIGN) - 1);
                } else {
                    lb = (s + 1) & ~(static_cast<int32_t>(CTPSV_FLOAT_ALIGN) - 1);
                }
            }
        }
        loadLen = ml + (mb - lb);
        if constexpr (UPLO == CtpsvUplo::UPPER && TRANS != CtpsvTrans::OP_N && DIAG == CtpsvDiag::NON_UNIT) {
            if (c + 1 == StepChunkCount(s)) {
                loadLen += 1; // 末块对角 = 装载末元素（stpsv FastLoadLen=dot+1 同构）
            }
        }
        loadBase = lb;
    }
    // chunk 装载起点（消费侧搭车元素定位：对角 elemIdx = s-LoadBase / 头段 elem0 = s0-LoadBase；
    // 仅 LOWER 步骤首调用，非热路径）
    __aicore__ inline int32_t LoadBase(int32_t s) const
    {
        if constexpr (DIAG == CtpsvDiag::NON_UNIT) {
            return s & ~(static_cast<int32_t>(CTPSV_FLOAT_ALIGN) - 1);
        } else {
            return (s + 1) & ~(static_cast<int32_t>(CTPSV_FLOAT_ALIGN) - 1);
        }
    }
    // 搭车 chunk0 合并消费（opt2 P0 + opt4 L3 头段窗口化，LOWER）：步骤首已 WaitColChunk；
    //   vred 自 srcOff=0 覆盖 前缀+主段（len' = ml + pref，pref ∈ {0,8}），头段元素并入
    //   向量主计算（垃圾 lane 处理：V1 = aR/aI [0,maskLen) 前置清零——Axpy 直写 b 平面，
    //   加法 ±0 保 NaN/Inf 语义；V2 = DotChunk 内 tR/tI [0,maskLen) 后置掩蔽——规避
    //   0×Inf=NaN 暴露面（文件头 opt2 P0「无 0·Inf 暴露面」设计点的 L3 承载形式））。
    //   maskLen = s0 - lb（垃圾 [lb,s0-1) + 对角 lane，NON_UNIT 对角已先行 col slot 直读）。
    __aicore__ inline void ConsumeColChunkMerged(const LocalTensor<float>& colIl, int32_t len, int32_t maskLen)
    {
        LocalTensor<float> aR = bufAR.Get<float>();
        LocalTensor<float> aI = bufAI.Get<float>();
        const int32_t lenA = CtpsvCeil8(len);
        const uint32_t rep = static_cast<uint32_t>((2 * lenA + 63) / 64);
        CtpsvVred(aR, colIl, rep, 1);
        CtpsvVred(aI, colIl, rep, 2);
        for (int32_t k = len; k < lenA; k++) { // dummy 尾填显式清零（≤7 元素/平面）
            aR.SetValue(k, 0.0f);
            aI.SetValue(k, 0.0f);
        }
        if constexpr (TRANS == CtpsvTrans::OP_N) {
            if (maskLen > 0) {         // V1：垃圾+对角 lane 前置清零（头段 lane [maskLen, pref) 保留）
                PipeBarrier<PIPE_V>(); // vred 写 → Duplicate 覆写序
                Duplicate(aR, 0.0f, static_cast<uint32_t>(maskLen));
                Duplicate(aI, 0.0f, static_cast<uint32_t>(maskLen));
            }
        }
        PipeBarrier<PIPE_V>();
        PipeBarrier<PIPE_ALL>();
    }

    // GM→UB 异步装载：AP 列 col 行段 [rowBase, rowBase+len) → col slot（packed 列内连续
    // 稠密段）。预取装载独立事件 id（MTE2_V_PF），装载完成由 ConsumeColChunk 等待。
    __aicore__ inline void IssueColChunk(int32_t col, int32_t base, int32_t len)
    {
        LocalTensor<float> colIl = (colSlot == 0) ? bufCol.Get<float>() : bufCol.Get<float>()[colFloats];
        DrainVBeforeColLoad(); // 目标 slot 上上一轮 vred 读排空后方可覆写（-O2 竞态纪律）
        const int64_t gmOff = 2LL * ColElemOff(col, base);
        DataCopyExtParams cp{1, 8U * static_cast<uint32_t>(len), 0, 0, 0};
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        DataCopyPad(colIl, apGM[gmOff], cp, pad);
        SetFlag<HardEvent::MTE2_V>(CTPSV_EVT_MTE2_V_PF);
        pfSlot = colSlot;
        if (dbuf) {
            colSlot ^= 1;
        }
    }

    // 等待在飞预取完成 + vreducev2 解交织 aR/aI 平面 + pad lane 显式清零（非搭车路径，
    // 形态与迭代三逐位一致——UPPER 全档 / LOWER c≥1 chunk / 搭车豁免尾步）。
    //  vred 越界读残片（装载 dummy 尾填/上一 chunk 残留）不进入消费域：计数形式
    //  Mul/Axpy 只消费 [0, lenA)、RRS 级联前 [lenA, lenR) 尾 lane 显式清零（文件头
    //  「迭代三 2)」三重护栏）——迭代二的 per-chunk Duplicate 全宽清零已移除（冗余
    //  V 带宽）。opt2 P0：wait 亦覆盖搭车元素标量读（LoadX STRIDED 先例）。
    __aicore__ inline void ConsumeColChunk(int32_t len)
    {
        LocalTensor<float> aR = bufAR.Get<float>();
        LocalTensor<float> aI = bufAI.Get<float>();
        WaitFlag<HardEvent::MTE2_V>(CTPSV_EVT_MTE2_V_PF);
        LocalTensor<float> colIl = (pfSlot == 0) ? bufCol.Get<float>() : bufCol.Get<float>()[colFloats];
        const int32_t lenA = CtpsvCeil8(len);
        const uint32_t rep = static_cast<uint32_t>((2 * lenA + 63) / 64);
        CtpsvVred(aR, colIl, rep, 1);
        CtpsvVred(aI, colIl, rep, 2);
        for (int32_t k = len; k < lenA; k++) { // dummy 尾填显式清零（≤7 元素/平面）
            aR.SetValue(k, 0.0f);
            aI.SetValue(k, 0.0f);
        }
        PipeBarrier<PIPE_V>();
        PipeBarrier<PIPE_ALL>();
    }

    // 非空步骤 s 的首 chunk 装载发射（chunk0 装载含搭车前缀/对角，ChunkLoad，opt2 P0）
    __aicore__ inline void IssueStepFirstChunk(int32_t s)
    {
        int32_t lb;
        int32_t ll;
        ChunkLoad(s, 0, lb, ll);
        IssueColChunk(s, lb, ll);
    }

    // 消费当前 chunk 后发射全局后继：同步骤下一 chunk，否则跨步 look-ahead 到后续
    // 首个非空步骤的首 chunk（空步序列仅出现于全局序列首/尾，扫描代价 O(1) 摊销）
    __aicore__ inline void IssueSuccessor(int32_t step, int32_t c)
    {
        if (c + 1 < StepChunkCount(step)) {
            int32_t lb;
            int32_t ll;
            ChunkLoad(step, c + 1, lb, ll);
            IssueColChunk(step, lb, ll);
            return;
        }
        int32_t s = step + (kFwd ? 1 : -1);
        while (s >= 0 && s < n) {
            if (StepChunkCount(s) > 0) {
                IssueStepFirstChunk(s);
                return;
            }
            s += (kFwd ? 1 : -1);
        }
    }

    // 流水序贯预取发射：全局序列首个 chunk（求解循环前；与 LoadX 尾部/首步标量段重叠）
    __aicore__ inline void IssueFirstChunk()
    {
        int32_t s = kFwd ? 0 : n - 1;
        while (s >= 0 && s < n) {
            if (StepChunkCount(s) > 0) {
                IssueStepFirstChunk(s);
                return;
            }
            s += (kFwd ? 1 : -1);
        }
    }

    // 搭车 chunk0 步骤首等待（opt2 P0，LOWER）：等待在飞预取完成（MTE2→V；标量读同受
    // 此 wait 覆盖——LoadX STRIDED / stpsv ComputeMainFast d=buf.GetValue(dot) 先例）
    __aicore__ inline void WaitColChunk() { WaitFlag<HardEvent::MTE2_V>(CTPSV_EVT_MTE2_V_PF); }

    // 当前在飞 chunk 的 col slot 视图（须已 WaitColChunk；pfSlot 未变期间有效）
    __aicore__ inline LocalTensor<float> ColSlotView()
    {
        return (pfSlot == 0) ? bufCol.Get<float>() : bufCol.Get<float>()[colFloats];
    }

    // V1 步骤 j 对角复除出 x_j：NON_UNIT 时 merged 步对角自 col slot 直读（前缀末元素），
    // 其余走独立 8B 微拷贝；UNIT 直通不读对角、无除法（spec diag 语义）
    __aicore__ inline void ColumnDivideXj(
        const float bjR, const float bjI, const bool merged, const int32_t j, float& xRe, float& xIm)
    {
        if constexpr (DIAG == CtpsvDiag::NON_UNIT) {
            float dr = 1.0f;
            float di = 0.0f;
            if (merged) {
                ReadDiagFromCol(ColSlotView(), j - LoadBase(j)); // 对角 = 前缀末元素
                dr = diagRe;
                di = diagIm;
            } else {
                LoadDiagScalar(j, dr, di); // T1 / 主段空尾步：独立 8B 微拷贝
            }
            ComplexDivide(bjR, bjI, dr, di, xRe, xIm);
        } else {
            xRe = bjR;
            xIm = bjI;
        }
    }

    // V1 merged 步 chunk0 消费：vred 自 srcOff=0 覆盖前缀+主段，Axpy 基回退 lb、长 +pref
    // （头段元素进宽向量；垃圾/对角 lane a=0 → b ± 0 保值）。有限性守卫：x_j 非有限时
    // 垃圾/对角 lane 的 alpha·0 = Inf·0 = NaN 会毒化已解行 [lb,j]，回退基线算术（头段
    // 标量 MAC col slot 直读 + 主段 Axpy 基 r0），Inf/NaN 传播语义与基线/golden 逐位
    // 一致。头段标量读须在 IssueSuccessor 前（单 slot 档 Issue 即 Drain+覆写）。
    __aicore__ inline void AxpyMergedChunk(
        const int32_t j, const int32_t c, const int32_t len, const float xRe, const float xIm)
    {
        const int32_t lb = LoadBase(j);
        const int32_t pref = StepSegStart(j) - lb;      // ∈ {0,8}（merged 步 mb<n）
        const LocalTensor<float> colIl = ColSlotView(); // Issue 前取视图（步首已 wait）
        ConsumeColChunkMerged(colIl, len + pref, (j + 1) - lb);
        const bool xFin = (xRe - xRe == 0.0f) && (xIm - xIm == 0.0f);
        if (!xFin) {
            HeadMacCore(colIl, (j + 1) - lb, j + 1, StepSegStart(j), xRe, xIm);
        }
        IssueSuccessor(j, c); // 后继装载与本 chunk Axpy 重叠（look-ahead）
        if (xFin) {
            TrailingAxpy(lb, len + pref, xRe, xIm);
        } else {
            TrailingAxpy(StepSegStart(j), len, xRe, xIm);
        }
    }

    // V1 非 merged chunk 消费：Consume → look-ahead → trailing Axpy。UPPER 档 AxpySeg
    // 计数 = ceil8(len) 使写域越出 trailing 段至 [j, ceil8(j))，pad lane 依赖 a_pad=0
    // 的「零贡献」保数值，但 IEEE 下 0 × Inf/NaN = NaN 会毒化已定稿元素；守卫：x_j
    // 非有限 → 计数取精确 len（pad lane 不写），有限路径计数 lenA 逐位不变。
    // LOWER constexpr 折叠零扰动。
    __aicore__ inline void AxpyNormalChunk(
        const int32_t j, const int32_t c, const int32_t base, const int32_t len, const float xRe, const float xIm)
    {
        ConsumeColChunk(len);
        IssueSuccessor(j, c); // 后继装载与本 chunk Axpy 重叠（look-ahead）
        if constexpr (UPLO == CtpsvUplo::UPPER) {
            const bool xFin = (xRe - xRe == 0.0f) && (xIm - xIm == 0.0f);
            TrailingAxpy(base, len, xRe, xIm, !xFin);
        } else {
            TrailingAxpy(base, len, xRe, xIm);
        }
    }

    // ── 形态 V1：列向 axpy right-looking（TRANS=OP_N；LOWER 前代升序 / UPPER 回代降序）──
    //    步骤 j 先出 x_j（NON_UNIT 对角标量复除：LOWER=列首 / UPPER=列末；UNIT 直通
    //    不读对角），再 trailing 更新 b -= AP_col·x_j：
    //      LOWER trailing = 行 (j, n)：头段 [j+1, r0) 标量复 MAC（r0=ceil8(j+1)，VEC
    //        子偏移对齐边界段，≤7 元素）+ 主段 [r0, n) 宽向量复 axpy 分 chunk；
    //      UPPER trailing = 行 [0, j)：前缀整段宽向量（起点 0 天然对齐）分 chunk。
    //    复 axpy 展开为双平面 4 条宽 Axpy（bR -= aR·xRe − aI·xIm；bI -= aR·xIm + aI·xRe）。
    //    迭代三：主段 chunk 走 look-ahead 预取流水（Consume → IssueSuccessor → Axpy）。
    //    迭代四 opt2 P0：LOWER 对角+头段搭车 chunk0 bulk 装载（对角=前缀末元素，步首
    //    col slot 直读）；UPPER（T1）对角在末块、步骤首即需（时序不可达）维持独立微拷贝。
    __aicore__ inline void SolveColumnAxpy()
    {
        constexpr bool FORWARD = (UPLO == CtpsvUplo::LOWER); // V1 ⇔ trans=OP_N
        LocalTensor<float> bAll = bufB.Get<float>();
        IssueFirstChunk();
        for (int32_t j = (FORWARD ? 0 : n - 1); (FORWARD ? j < n : j >= 0); (FORWARD ? ++j : --j)) {
            SetFlag<HardEvent::V_S>(CTPSV_EVT_V_S); // 上步向量写（Axpy trailing）→ 本步标量读 b_j
            WaitFlag<HardEvent::V_S>(CTPSV_EVT_V_S);
            const float bjR = bAll.GetValue(j);
            const float bjI = bAll.GetValue(iPlane + j);
            constexpr bool kMergeLower = (UPLO == CtpsvUplo::LOWER); // 常量前置：UPPER 模板
            //  上编译期折叠为 false（bisheng 二进制恒等对照实证已折叠；留档注释）
            int32_t cnt = 0;
            if constexpr (kMergeLower) {
                cnt = StepChunkCount(j); // LOWER 步首（搭车判定；UPPER 保持原位求值）
            }
            const bool merged = kMergeLower && (cnt > 0); // 搭车装载（LOWER）
            if (merged) {
                WaitColChunk(); // chunk0 搭车装载到位（头段/对角标量读前置条件）
            }
            float xRe;
            float xIm;
            ColumnDivideXj(bjR, bjI, merged, j, xRe, xIm);
            bAll.SetValue(j, xRe);
            bAll.SetValue(iPlane + j, xIm);
            SetFlag<HardEvent::S_V>(CTPSV_EVT_S_V); // S 写 x_j → 后续 V Axpy 可见（含跨步 trailing 读）
            WaitFlag<HardEvent::S_V>(CTPSV_EVT_S_V);
            if constexpr (!kMergeLower) {
                cnt = StepChunkCount(j); // UPPER 原位求值（迭代三控制序逐位保持）
            }

            if constexpr (UPLO == CtpsvUplo::LOWER) {
                // merged 步头段 [j+1, r0) 已并入 chunk0 宽向量 Axpy；仅主段空尾步（cnt=0）
                // 走遗留 bufDiag 标量微拷贝路径
                if (!merged) {
                    const int32_t s0 = j + 1;
                    const int32_t r0 = (CtpsvCeil8(s0) < n) ? CtpsvCeil8(s0) : n;
                    ApplyHeadMac(j, s0, r0, xRe, xIm);
                }
            }
            for (int32_t c = 0; c < cnt; ++c) {
                const int32_t base = StepSegStart(j) + c * chunkLen;          // 主段基（内联）
                const int32_t len = CtpsvMin(chunkLen, StepSegEnd(j) - base); // 禁越界读
                if (c == 0 && merged) {
                    AxpyMergedChunk(j, c, len, xRe, xIm);
                } else {
                    AxpyNormalChunk(j, c, base, len, xRe, xIm);
                }
            }
        }
        PipeBarrier<PIPE_ALL>(); // S/V 写 → 写回（Gather/DataCopyPad）序收口
    }

    // ══════════════════════════════════════════════════════════════════════
    // opt5 r2 杆 4：V1 BLK2（B2LP）—— LOWER/N/UNIT（T8）前代对 (j, j+1) 单迭代
    // （probe_blk2_v1 TU 分离探针真机验证后上主线：corr 114/114（G1+G2+同序
    //  模型）+ perf 全带 −14~16% + Inf 域 A/B vs golden 不可判别留档（复数
    //  Inf soup 塌缩，回退路径门域外））。行对结构：
    //    窗口1 = 列 j 行 [lb, n)（lb = j&~7，主线 T8 UNIT 步 j LoadBase 同式
    //            (j+1)&~7 ≡ j&~7（j 偶））：lanes [lb,j) 前列尾垃圾 / lane j
    //            对角（UNIT 无视）/ lane j+1 = A[j+1,j] 闭式尾项（标量读出
    //            lane tailIdx = j+1−lb ∈ {1,3,5,7}）/ lanes [j+2,n) trailing；
    //    窗口2 = 列 j+1 行 [lb2, n)（lb2 = (j+2)&~7，主线步 j+1 LoadBase 同式）
    //            ——lanes [lb2,j+1) 垃圾 / lane j+1 对角 / lanes [j+2,n) trailing。
    //  两窗掩蔽域 [lb, j+2) / [lb2, j+2)（V1 a 平面前置清零，b±0 保值——
    //  对角/闭式 lane 的 b 行经 S_V 对先行落值）；trailing 行区间两列同为
    //  [j+2, n)。⚠ 单装载（两窗物理并域）不可行已证：两窗距 ≡ n−j−1 (mod 4)
    //  随对递进交替（j += 2），vred src 32B / Axpy dst 32B 双对齐无解——故取
    //  双窗双 slot 双 DMA 形态（ctrsv B2LP 双列装载 + 杆 2 双 slot 结构）。
    //  每对省 V_S/S_V/PF wait/DrainV 各 1 组（2→1）+ 步首结构减半；DMA/vred/
    //  Axpy 向量工作量不变（L1 调度锚负知识域——S_V 对减半非删除，探针真机
    //  全带反向筛查通过）。
    //  闭式（UNIT，非位级——x_{j+1} 二项闭式 vs 主线向量 lane 三项累加，舍入
    //  序差异；CPU 同序模型交叉验证 + 全精度门承载）：x_j = b_j；
    //  x_{j+1} = b_{j+1} − A[j+1,j]·x_j。
    //  有限性守卫（fill_x_inf 域，按对重推导）：x_j 非有限 → 列 j 走头段标量
    //  MAC（HeadMacCore col slot 直读）+ Axpy 基 r0 src 偏移 pref（回退路径
    //  行映射 = aR[k] ↔ row lb+k，pref = r0−lb ∈ {0,8} 恒 32B 对齐）；x_{j+1}
    //  非有限 → 列 j+1 同式（掩蔽 lane 0·Inf = NaN 防护）。奇 n 末单元素
    //  j = n−1：列 ≤ n−3 trailing 已更新 b_{n−1}，UNIT 直通即终值零写回。
    // ══════════════════════════════════════════════════════════════════════

    // BLK2 V1 trailing（src 偏移形态）：b[base..] −= aCol·x，a 平面 lane 偏移
    // srcOff（窗口基 lb → 消费行 base 的行差，∈ {0,8} 恒 32B 对齐——回退路径
    // 正确语序：aR[k] ↔ row lb+k）
    __aicore__ inline void TrailingAxpyOff(int32_t base, int32_t len, int32_t srcOff, float xRe, float xIm)
    {
        LocalTensor<float> bAll = bufB.Get<float>();
        LocalTensor<float> aR = bufAR.Get<float>();
        LocalTensor<float> aI = bufAI.Get<float>();
        SetMaskCount();
        AxpySeg(bAll[base], aR[srcOff], -xRe, len);          // bR -= aR·xRe
        AxpySeg(bAll[base], aI[srcOff], xIm, len);           // bR += aI·xIm
        AxpySeg(bAll[iPlane + base], aR[srcOff], -xIm, len); // bI -= aR·xIm
        AxpySeg(bAll[iPlane + base], aI[srcOff], -xRe, len); // bI -= aI·xRe
        SetMaskNorm();
        ResetMask();
        PipeBarrier<PIPE_V>();
    }

    // V1 BLK2 循环体分段（SolveColumnBlk2 拆分，语序逐段保持）：
    //   列 j 窗口消费 + 闭式存储（Col1Head/StorePair）→ 列 j trailing（Col1Trailing）
    //   → 窗口2消费 + 后继发射 + 列 j+1 trailing（Col2Trailing）。
    // 列 j 步首：V_S 对 + 窗口1 等待 + 闭式尾项标量读 + b_j 读 + !xFin1 头段标量 MAC
    // （须在后继窗1 覆写 slot0 前）+ 窗口1 vred 消费 + 后继对窗1 发射
    __aicore__ inline void Blk2AxpyCol1Head(int32_t j, float& xjR, float& xjI, bool& xFin1, float& aTR, float& aTI)
    {
        LocalTensor<float> bAll = bufB.Get<float>();
        const int32_t lb = j & ~(static_cast<int32_t>(CTPSV_FLOAT_ALIGN) - 1);
        const int32_t lb2 = (j + 2) & ~(static_cast<int32_t>(CTPSV_FLOAT_ALIGN) - 1);
        const int32_t tailIdx = (j + 1) - lb; // A[j+1,j] = 窗口1 lane（行 j+1）
        const bool nextPair = (j + 3 < n);
        const bool hasTail = (j + 2 < n);       // trailing 行区间 [j+2,n) 非空
        SetFlag<HardEvent::V_S>(CTPSV_EVT_V_S); // 上对 trailing 向量写 → 本对标量读
        WaitFlag<HardEvent::V_S>(CTPSV_EVT_V_S);
        // 窗口1 等待 + 闭式尾项标量读（MTE2_V_PF wait 覆盖标量读）
        WaitFlag<HardEvent::MTE2_V>(CTPSV_EVT_MTE2_V_PF);
        LocalTensor<float> colIl0 = bufCol.Get<float>();
        aTR = colIl0.GetValue(2 * tailIdx);
        aTI = colIl0.GetValue(2 * tailIdx + 1);
        xjR = bAll.GetValue(j);
        xjI = bAll.GetValue(iPlane + j);
        xFin1 = (xjR - xjR == 0.0f) && (xjI - xjI == 0.0f); // 有限性守卫（v−v==0 ⇔ 有限）
        if (!xFin1) {
            bAll.SetValue(j, xjR);
            bAll.SetValue(iPlane + j, xjI);
            HeadMacCore(colIl0, tailIdx, j + 1, (CtpsvCeil8(j + 1) < n) ? CtpsvCeil8(j + 1) : n, xjR, xjI);
        }
        if (hasTail) {
            ConsumeColChunkMerged(colIl0, n - lb, j + 2 - lb); // 掩蔽域 [lb, j+2)
        }
        if (nextPair) {
            DrainVBeforeColLoad();
            IssueWindow(j + 2, lb2, 0);
            SetFlag<HardEvent::MTE2_V>(CTPSV_EVT_MTE2_V_PF);
        }
    }

    // 闭式（UNIT）+ 成对落值 + S_V 对：x_{j+1} = b_{j+1} − A[j+1,j]·x_j（x_j 有限）/
    // 头段 MAC 后直读（!xFin1，b_{j+1} 已被头段更新）
    __aicore__ inline void Blk2AxpyStorePair(
        int32_t j, const float xjR, const float xjI, const bool xFin1, const float aTR, const float aTI, float& xmR,
        float& xmI, bool& xFin2)
    {
        LocalTensor<float> bAll = bufB.Get<float>();
        const float bmR = bAll.GetValue(j + 1);
        const float bmI = bAll.GetValue(iPlane + j + 1);
        if (xFin1) {
            xmR = bmR - (aTR * xjR - aTI * xjI);
            xmI = bmI - (aTR * xjI + aTI * xjR);
        } else {
            xmR = bmR;
            xmI = bmI;
        }
        xFin2 = (xmR - xmR == 0.0f) && (xmI - xmI == 0.0f);
        if (xFin1) {
            bAll.SetValue(j, xjR);
            bAll.SetValue(iPlane + j, xjI);
        }
        bAll.SetValue(j + 1, xmR);
        bAll.SetValue(iPlane + j + 1, xmI);
        SetFlag<HardEvent::S_V>(CTPSV_EVT_S_V); // S 写 x_j/x_{j+1} → 掩蔽 lane b±0 读 + trailing 读
        WaitFlag<HardEvent::S_V>(CTPSV_EVT_S_V);
    }

    // 列 j trailing：xFin → 掩蔽 lane a=0 → b±0 保值；!xFin → 回退基 r0 + src 偏移
    // （正确行映射语序）
    __aicore__ inline void Blk2AxpyCol1Trailing(int32_t j, const float xjR, const float xjI, const bool xFin1)
    {
        if (j + 2 >= n) {
            return; // trailing 行区间 [j+2,n) 空
        }
        const int32_t lb = j & ~(static_cast<int32_t>(CTPSV_FLOAT_ALIGN) - 1);
        const int32_t r0Raw = CtpsvCeil8(j + 1);
        const int32_t r0 = (r0Raw < n) ? r0Raw : n;
        if (xFin1) {
            TrailingAxpy(lb, n - lb, xjR, xjI);
        } else {
            TrailingAxpyOff(r0, n - r0, r0 - lb, xjR, xjI);
        }
    }

    // 窗口2 等待 + vred + 后继对窗2发射 + 列 j+1 trailing（xFin2 守卫独立于列 j——
    // 掩蔽 lane 0·x_{j+1}）
    __aicore__ inline void Blk2AxpyCol2Trailing(int32_t j, const float xmR, const float xmI, const bool xFin2)
    {
        const int32_t lb2 = (j + 2) & ~(static_cast<int32_t>(CTPSV_FLOAT_ALIGN) - 1);
        const bool nextPair = (j + 3 < n);
        const bool hasTail = (j + 2 < n);
        WaitFlag<HardEvent::MTE2_V>(CTPSV_EVT_MTE2_V_PF2); // flag 卫生：无条件 wait
        PipeBarrier<PIPE_V>();
        if (hasTail) { // hasTail ⟹ lb2 ≤ j+2 < n（窗口非空）；尾对免 vred
            LocalTensor<float> colIl = bufCol.Get<float>();
            ConsumeColChunkMerged(colIl[colFloats], n - lb2, j + 2 - lb2);
        }
        if (nextPair) {
            const int32_t lb2N = (j + 4) & ~(static_cast<int32_t>(CTPSV_FLOAT_ALIGN) - 1);
            DrainVBeforeColLoad();
            IssueWindow(j + 3, lb2N, 1);
            SetFlag<HardEvent::MTE2_V>(CTPSV_EVT_MTE2_V_PF2);
        }
        if (hasTail) {
            const int32_t r0Raw = CtpsvCeil8(j + 1);
            const int32_t r0 = (r0Raw < n) ? r0Raw : n;
            if (xFin2) {
                TrailingAxpy(lb2, n - lb2, xmR, xmI);
            } else {
                TrailingAxpyOff(r0, n - r0, r0 - lb2, xmR, xmI);
            }
        }
    }

    __aicore__ inline void SolveColumnBlk2()
    {
        if (n <= 1) {
            PipeBarrier<PIPE_ALL>();
            return; // x_0 = b_0（UNIT，空 trailing，b 平面原值即解）
        }
        // prologue：首对 (0,1) 双窗装载（首对 lb = lb2 = 0；PF1/PF2 独立 id 各一次 set）
        {
            DrainVBeforeColLoad();
            IssueWindow(0, 0, 0);
            SetFlag<HardEvent::MTE2_V>(CTPSV_EVT_MTE2_V_PF);
            IssueWindow(1, 0, 1);
            SetFlag<HardEvent::MTE2_V>(CTPSV_EVT_MTE2_V_PF2);
        }
        for (int32_t j = 0; j + 1 < n; j += 2) {
            float xjR;
            float xjI;
            float xmR;
            float xmI;
            float aTR;
            float aTI;
            bool xFin1;
            bool xFin2;
            Blk2AxpyCol1Head(j, xjR, xjI, xFin1, aTR, aTI);
            Blk2AxpyStorePair(j, xjR, xjI, xFin1, aTR, aTI, xmR, xmI, xFin2);
            Blk2AxpyCol1Trailing(j, xjR, xjI, xFin1);
            Blk2AxpyCol2Trailing(j, xmR, xmI, xFin2);
        }
        // 奇 n 末单元素 j = n−1：列 ≤ n−3 trailing 已更新 b_{n−1}，UNIT 直通即
        // 终值（b 平面原值即解，零写回）
        PipeBarrier<PIPE_ALL>(); // S/V 写 → 写回（Gather/DataCopyPad）序收口
    }

    // V2 dot 缓冲入口预清零（tR/tI 尾垫 + w 二级读域恒零垫，SolveRowDot /
    // SolveRowDotBlk2 共用；位级等价论证见两调用点注释）
    __aicore__ inline void PreZeroDotBuffers()
    {
        LocalTensor<float> tR = bufTR.Get<float>();
        LocalTensor<float> tI = bufTI.Get<float>();
        LocalTensor<float> w = bufW.Get<float>();
        const uint32_t zWidth = static_cast<uint32_t>(CtpsvCeil64(chunkPad) + planeRide);
        Duplicate(tR, 0.0f, zWidth);
        Duplicate(tI, 0.0f, zWidth);
        Duplicate(w, 0.0f, 2U * CTPSV_W_SPAN);
        PipeBarrier<PIPE_V>();
    }

    // ── 形态 V2：左视列 dot（TRANS=OP_T/OP_C；T3-T6 UPPER 前代 / T9-T12 LOWER 回代）──
    //    left-looking：步骤 i 装 AP 列 i 已解段（UPPER 前缀 [0,i) / LOWER 尾段 (i,n)，
    //    packed 列内连续稠密段，访存集合与 V1 同），dot s = Σ T[i,已解]·x_已解
    //    （解交织 4 Mul + Add/Sub×2 逐元素部分和 → RRS 级联全和，全程 UB 内）；
    //    对角复除（OP_C 为 conj(d)，invIm 符号位编译期翻转）；UNIT 免除法不读对角。
    //    迭代三：chunk 走 look-ahead 预取流水（Consume → IssueSuccessor → DotChunk）。
    //    迭代四 opt2 P0：LOWER 头段+对角搭车 chunk0 bulk 装载（步首 col slot 直读，头段
    //    标量累加序不变 = bit-exact）；UPPER 末块装载长 +1（对角 = 末元素，stpsv 同构），
    //    对角 lane 落入 [len,lenA) 时既有 pad 清零护栏覆盖。
    // V2 步骤首 LOWER 头段/对角处理：merged 步头段 [i+1, r0) 已并入 chunk0 宽向量
    // dot（DotChunk 内 tR/tI 后置掩蔽），仅等待搭车装载并对角 col slot 步首直读；
    // 主段空尾步（cnt=0）走遗留 bufDiag 标量微拷贝路径
    __aicore__ inline void RowDotStepHead(const int32_t i, const bool merged, float& sR, float& sI)
    {
        if constexpr (UPLO == CtpsvUplo::LOWER) {
            if (!merged) {
                const int32_t s0 = i + 1;
                const int32_t r0 = (CtpsvCeil8(s0) < n) ? CtpsvCeil8(s0) : n;
                ApplyHeadDot(i, s0, r0, sR, sI);
            } else {
                WaitColChunk(); // chunk0 搭车装载到位
                if constexpr (DIAG == CtpsvDiag::NON_UNIT) {
                    ReadDiagFromCol(ColSlotView(), i - LoadBase(i)); // 对角 = 前缀末元素
                }
            }
        }
    }

    // V2 merged 步 chunk0 消费：vred 自 srcOff=0 覆盖前缀+主段，dot 基回退 lb、长
    // +pref（头段元素进宽向量；垃圾/对角 lane t 后置掩蔽防 0×Inf）
    __aicore__ inline void DotChunkMergedStep(
        const int32_t i, const int32_t c, const int32_t len, const bool monoZero, float& sR, float& sI)
    {
        const int32_t lb = LoadBase(i);
        const int32_t pref = StepSegStart(i) - lb;                      // ∈ {0,8}（merged 步 mb<n）
        ConsumeColChunkMerged(ColSlotView(), len + pref, (i + 1) - lb); // 步首已 wait
        IssueSuccessor(i, c); // 后继装载与本 chunk dot 重叠（look-ahead）
        DotChunk(lb, len + pref, (i + 1) - lb, sR, sI, monoZero);
    }

    // V2 非 merged chunk 消费：Consume → 末块对角直读（UPPER dot +1 搭车档，读出后
    // pad 清零护栏覆盖其 lane）→ look-ahead → DotChunk
    __aicore__ inline void DotChunkNormalStep(
        const int32_t i, const int32_t c, const int32_t base, const int32_t len, const int32_t cnt, const bool monoZero,
        float& sR, float& sI)
    {
        ConsumeColChunk(len);
        if constexpr (UPLO == CtpsvUplo::UPPER && TRANS != CtpsvTrans::OP_N && DIAG == CtpsvDiag::NON_UNIT) {
            if (c + 1 == cnt) {
                ReadDiagFromCol(ColSlotView(), len); // 末块对角 = 装载末元素
            }
        }
        IssueSuccessor(i, c); // 后继装载与本 chunk dot 重叠（look-ahead）
        DotChunk(base, len, 0, sR, sI, monoZero);
    }

    // V2 步骤尾对角复除（NON_UNIT）：主段空尾步 diagRe/diagIm 独立 8B 微拷贝后
    // 复除；搭车档 diagRe/diagIm 已于步骤首读出。UNIT 无除法不读对角
    __aicore__ inline void RowDotDivideX(const int32_t i, const int32_t cnt, float& pR, float& pI)
    {
        if constexpr (DIAG == CtpsvDiag::NON_UNIT) {
            if (cnt == 0) {
                LoadDiagScalar(i, diagRe, diagIm); // 尾步遗留（UPPER i=0 / LOWER 主段空）
            }
            ComplexDivide(pR, pI, diagRe, diagIm, pR, pI);
        }
    }

    __aicore__ inline void SolveRowDot()
    {
        constexpr bool FORWARD = (UPLO == CtpsvUplo::UPPER); // V2：(U,T)/(U,C) 前代、(L,T)/(L,C) 回代
        LocalTensor<float> bAll = bufB.Get<float>();
        // opt5 杆 1（stpsv 单调免清零）：tR/tI/w/w2 一次预清零（stpsv IssueMainFast
        //   Init 清零 + 单调免重清同域先例）。位级等价论证：
        //   ① tR/tI 尾垫 [lenA,lenR)：count 形式 Mul/Add/Sub 只写 [0,lenA)、掩蔽
        //     Duplicate 只写 [0,maskLen≤8≤lenA) → lane≥lenA 者仅当被更早步写过才非零；
        //     整列单块档（monoZero）lenA 逐步非降 → 尾垫 lane 从未被写 → Init 零恒存；
        //   ② w 二级读域 [rep,64)/[64+rep,128)：一级 RRS 只写 [0,rep)∪[64,64+rep)，
        //     rep 逐步非降 → 域外 lane 从未被写 → Init 零恒存（w2 消费域 [0,2) 恒被
        //     本步二级覆写，无跨步读）；
        //   ③ 多 chunk 档（monoZero=false）：每步 Duplicate 清零照旧覆写，预清零仅
        //     为冗余前置，写值/写域/消费域与 opt4 终态逐位一致。
        //   谓词求值于 V2 专属函数局部（非类成员/Init）——V1 模板（T1/T2/T7/T8）不
        //   实例化 SolveRowDot/DotChunk，类布局与 Init codegen 逐位保持（布局税纪律）。
        //   整列单块 ⇔ chunkLen==n（host 策略 ⇔ FAST n≤4096 / GATHER·STRIDED n≤2048）：
        //   LOWER 回代 len'=n-lb（lb=(i+1 或 i)&~7 随 i 降非增）/ UPPER 前代 len=i
        //   （随 i 升非降）→ lenA/rep 逐步非降；多 chunk 档 chunk0 的 len'=chunkLen+pref
        //   在组界 8↔0 回摆非单调 → false 走每步清零遗留路径。
        const bool monoZero = (chunkLen >= n);
        PreZeroDotBuffers();
        IssueFirstChunk();
        for (int32_t i = (FORWARD ? 0 : n - 1); (FORWARD ? i < n : i >= 0); (FORWARD ? ++i : --i)) {
            PipeBarrier<PIPE_V>(); // 上步向量写（Mul 级联）→ 本步标量读 b_i（V 内按序收口）
            float sR = 0.0f, sI = 0.0f;
            const int32_t cnt = StepChunkCount(i);
            constexpr bool kMergeLower = (UPLO == CtpsvUplo::LOWER);
            const bool merged = kMergeLower && (cnt > 0); // 头段搭车（LOWER）
            RowDotStepHead(i, merged, sR, sI);
            for (int32_t c = 0; c < cnt; ++c) {
                const int32_t base = StepSegStart(i) + c * chunkLen;          // 主段基（内联）
                const int32_t len = CtpsvMin(chunkLen, StepSegEnd(i) - base); // 禁越界读
                if (c == 0 && merged) {
                    DotChunkMergedStep(i, c, len, monoZero, sR, sI);
                } else {
                    DotChunkNormalStep(i, c, base, len, cnt, monoZero, sR, sI);
                }
            }
            const float biR = bAll.GetValue(i);
            const float biI = bAll.GetValue(iPlane + i);
            float pR = biR - sR;
            float pI = biI - sI;
            RowDotDivideX(i, cnt, pR, pI);
            bAll.SetValue(i, pR);
            bAll.SetValue(iPlane + i, pI);
            SetFlag<HardEvent::S_V>(CTPSV_EVT_S_V); // S 写 x_i → 后续 V Mul 可见（下步 dot 消费）
            WaitFlag<HardEvent::S_V>(CTPSV_EVT_S_V);
        }
        PipeBarrier<PIPE_ALL>(); // S/V 写 → 写回（Gather/DataCopyPad）序收口
    }

    // ══════════════════════════════════════════════════════════════════════
    // opt5 杆 2：B-lite BLK2 —— V2 LOWER T/C UNIT 两步并一迭代（TU 分离探针
    // probe_blk2_v2 真机验证后上主线；228/228 G1+G2+同序模型全绿 + A/B 收益实测）。
    //   行对 (i, i−1)（i 奇，iTop 起 −2 递减）单迭代：
    //     窗口1 = 列 i−1 行 [lb, n)、窗口2 = 列 i 行 [lb, n)，lb = i & ~7（≤ i 恒
    //     成立——窗1 须含尾项行 i；遗留 x_{i−1} 步 LoadBase UNIT 同式）；
    //     双窗装载双 DMA 双独立 PF id（3/4），两窗各自 vred 消费后即发射后继对同位
    //     窗（fixA：覆盖 ≈70~90% 对长——pair 尾发射实测边际等待 +2~3% 劣化，探针
    //     定谳）；两窗同 b 切片 bAll[lb..n)，
    //     掩蔽域同 [lb, i]（maskLen = i+1−lb ∈ [1,8]，垃圾/对角/尾项 lane 乘后掩 t）；
    //     dot 段两列同为行 [i+1, n)——x_i 不进任何向量 MAC。
    //   闭式（UNIT，非位级——尾项移出 RRS 树后置标量减，CPU 同序模型交叉验证）：
    //     x_i = b_i − s_i；x_{i−1} = b_{i−1} − s'_{i−1} − A'·x_i
    //     （A' = A[i,i−1] / OP_C: conj(A[i,i−1])；标量读自窗1 col slot lane i−lb，
    //      掩蔽只清 tR/tI，slot 数据不动——ReadDiagFromCol 同构先例）。
    //   每对省 1 DrainV + 1 PF wait 对 + 1 S_V 对 + 步首 barrier + 半数标量开销；
    //   lenA = n−lb 两 pass 同值且逐对非降 → 杆 1 预清零恒零垫论证同域成立。
    //   UB：复用 dbuf 双 col slot + 既有全部平面，零增量零 host 守卫变更。
    // ══════════════════════════════════════════════════════════════════════

    // BLK2 窗口装载发射：列 col 行段 [lb, n) → slot（0/1）；调用前须已 DrainVBeforeColLoad
    __aicore__ inline void IssueWindow(int32_t col, int32_t lb, int32_t slotIdx)
    {
        LocalTensor<float> colIl = bufCol.Get<float>();
        LocalTensor<float> dst = (slotIdx == 0) ? colIl : colIl[colFloats];
        const int64_t gmOff = 2LL * ColElemOff(col, lb);
        DataCopyExtParams cp{1, 8U * static_cast<uint32_t>(n - lb), 0, 0, 0};
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        DataCopyPad(dst, apGM[gmOff], cp, pad);
    }

    // BLK2 窗口 vred 消费：aR/aI ← slot（行 [lb, n) 段）。opt5 r2 杆 4b：dummy
    //   尾 [len, lenA) 标量清零环删除——DotWindowL1 计数收紧至 len（count 形式
    //   set_mask_count+set_vector_mask(0,count) 精确元素界，dav_c220 impl 实证），
    //   a 尾垃圾 lane 不再进入任何向量 MAC（0×Inf=NaN 暴露面一并封闭）；t 尾垫
    //   [len, lenA) 恒为入口预清零零值（len 逐对非降，杆 1 论证同域）→ 位级等价。
    __aicore__ inline void ConsumeWindow(int32_t slotIdx, int32_t lb)
    {
        LocalTensor<float> colIl = bufCol.Get<float>();
        LocalTensor<float> aR = bufAR.Get<float>();
        LocalTensor<float> aI = bufAI.Get<float>();
        LocalTensor<float> src = (slotIdx == 0) ? colIl : colIl[colFloats];
        const int32_t len = n - lb;
        const int32_t lenA = CtpsvCeil8(len);
        const uint32_t rep = static_cast<uint32_t>((2 * lenA + 63) / 64);
        CtpsvVred(aR, src, rep, 1);
        CtpsvVred(aI, src, rep, 2);
        PipeBarrier<PIPE_V>();
        PipeBarrier<PIPE_ALL>();
    }

    // 杆 3：融合级联 level-2 + 单 V_S（4 槽 w[0..256)：窗1 sR/sI 部分和落
    // [0..rep)∪[64..64+rep)、窗2 落 [128..128+rep)∪[192..192+rep)；二级单趟
    // 4 repeat → w2[0..4)=[s1R,s1I,s2R,s2I]——与两趟 CascadeSum2 逐组件树同构
    // → 位级一致（vs 杆 2 形态）；每对省 1 条二级 RRS + 1 组 V_S 对 + 2 GetValue。
    //   w[0..256) Init 预清零 + rep 两 pass 同值逐对非降 → 槽间 [rep,64) 垫恒零。
    __aicore__ inline void CascadeSum4(
        const LocalTensor<float>& w, const LocalTensor<float>& w2, float& s1R, float& s1I, float& s2R, float& s2I)
    {
        CtpsvMaskFull();
        RepeatReduceSum<float, false>(w2, w, 4, 0, 0, 1, 1, 8);
        SetFlag<HardEvent::V_S>(CTPSV_EVT_V_S);
        WaitFlag<HardEvent::V_S>(CTPSV_EVT_V_S);
        s1R = w2.GetValue(0);
        s1I = w2.GetValue(1);
        s2R = w2.GetValue(2);
        s2I = w2.GetValue(3);
    }

    // BLK2 单窗口 MAC（DotChunk merged 形态镜像；入口预清零 + lenA 逐步非降 → 尾垫
    // 免清零，掩蔽 Duplicate 保留）：level-1 部分和落 w[wBase]/w[wBase+64]。
    //   opt5 r2 杆 4b：6 条向量计数收紧 lenA→len（精确 count 界）——a 尾垃圾
    //   lane [len, lenA) 不进 MAC；t 尾垫恒零（Init 预清零 + len 逐对非降，
    //   杆 1 同域）→ RRS 消费域值逐位一致（位级等价 vs 杆 3 形态）
    __aicore__ inline void DotWindowL1(int32_t lb, int32_t maskLen, int32_t wBase)
    {
        LocalTensor<float> bAll = bufB.Get<float>();
        LocalTensor<float> aR = bufAR.Get<float>();
        LocalTensor<float> aI = bufAI.Get<float>();
        LocalTensor<float> tR = bufTR.Get<float>();
        LocalTensor<float> tI = bufTI.Get<float>();
        LocalTensor<float> t2 = bufT2.Get<float>();
        LocalTensor<float> w = bufW.Get<float>();
        LocalTensor<float> w2 = w[CTPSV_W_SPAN];
        const int32_t len = n - lb;
        const int32_t lenA = CtpsvCeil8(len);
        const LocalTensor<float> bR = bAll[lb];
        const LocalTensor<float> bI = bAll[iPlane + lb];
        Mul(tR, aR, bR, static_cast<uint32_t>(len));
        PipeBarrier<PIPE_V>();
        Mul(t2, aI, bI, static_cast<uint32_t>(len));
        PipeBarrier<PIPE_V>();
        if constexpr (TRANS == CtpsvTrans::OP_C) {
            Add(tR, tR, t2, static_cast<uint32_t>(len));
        } else {
            Sub(tR, tR, t2, static_cast<uint32_t>(len));
        }
        PipeBarrier<PIPE_V>();
        Mul(tI, aR, bI, static_cast<uint32_t>(len));
        PipeBarrier<PIPE_V>();
        Mul(t2, aI, bR, static_cast<uint32_t>(len));
        PipeBarrier<PIPE_V>();
        if constexpr (TRANS == CtpsvTrans::OP_C) {
            Sub(tI, tI, t2, static_cast<uint32_t>(len));
        } else {
            Add(tI, tI, t2, static_cast<uint32_t>(len));
        }
        PipeBarrier<PIPE_V>();
        if (maskLen > 0) { // L3 后置掩蔽（垃圾 lane × 未解 b 可能 Inf/NaN，乘后掩 t）
            Duplicate(tR, 0.0f, static_cast<uint32_t>(maskLen));
            Duplicate(tI, 0.0f, static_cast<uint32_t>(maskLen));
            PipeBarrier<PIPE_V>();
        }
        CtpsvMaskFull();
        const int32_t rep = (lenA + 63) >> 6;
        RepeatReduceSum<float, false>(w[wBase], tR, static_cast<uint8_t>(rep), 0, 0, 1, 1, 8);
        RepeatReduceSum<float, false>(w[wBase + 64], tI, static_cast<uint8_t>(rep), 0, 0, 1, 1, 8);
        PipeBarrier<PIPE_V>();
    }

    // V2 BLK2 循环体分段（SolveRowDotBlk2 拆分，语序逐段保持）。
    // 窗口1（列 i−1）：等待 + 尾项标量读 + vred 消费 + 后继对窗1发射（MTE2 与本对
    // MAC 重叠）；A[i,i−1] = 窗口1 lane（行 i）
    __aicore__ inline void Blk2DotWindow1(const int32_t i, const int32_t lb, float& aTR, float& aTI)
    {
        WaitFlag<HardEvent::MTE2_V>(CTPSV_EVT_MTE2_V_PF);
        LocalTensor<float> colIl = bufCol.Get<float>();
        const int32_t tailIdx = i - lb;
        aTR = colIl.GetValue(2 * tailIdx);
        aTI = colIl.GetValue(2 * tailIdx + 1);
        ConsumeWindow(0, lb);
        if (i >= 3) { // 后继对 (i−2, i−3) 窗1 = 列 i−3
            const int32_t lbN = (i - 2) & ~(static_cast<int32_t>(CTPSV_FLOAT_ALIGN) - 1);
            DrainVBeforeColLoad();
            IssueWindow(i - 3, lbN, 0);
            SetFlag<HardEvent::MTE2_V>(CTPSV_EVT_MTE2_V_PF);
        }
    }

    // 窗口2（列 i）：等待 + vred 消费（覆盖 aR/aI，WAR 由 V 序 + barrier 序）+ 后继
    // 对窗2即时发射（slot1 已消费，MTE2 覆盖本对后续计算段）
    __aicore__ inline void Blk2DotWindow2(const int32_t i, const int32_t lb)
    {
        WaitFlag<HardEvent::MTE2_V>(CTPSV_EVT_MTE2_V_PF2);
        PipeBarrier<PIPE_V>();
        ConsumeWindow(1, lb);
        if (i >= 3) { // 后继对 (i−2, i−3) 窗2 = 列 i−2
            const int32_t lbN = (i - 2) & ~(static_cast<int32_t>(CTPSV_FLOAT_ALIGN) - 1);
            DrainVBeforeColLoad();
            IssueWindow(i - 2, lbN, 1);
            SetFlag<HardEvent::MTE2_V>(CTPSV_EVT_MTE2_V_PF2);
        }
    }

    // 闭式（UNIT）落值 + S_V 对：x_i = b_i − s_i；x_{i−1} = b_{i−1} − s' − A'·x_i
    // （OP_C：A' = conj(A[i,i−1])，符号位编译期翻转）
    __aicore__ inline void Blk2DotStorePair(
        const int32_t i, const float aTR, const float aTI, const float s1R, const float s1I, const float s2R,
        const float s2I)
    {
        LocalTensor<float> bAll = bufB.Get<float>();
        const float biR = bAll.GetValue(i);
        const float biI = bAll.GetValue(iPlane + i);
        const float bmR = bAll.GetValue(i - 1);
        const float bmI = bAll.GetValue(iPlane + i - 1);
        const float xIR = biR - s2R;
        const float xII = biI - s2I;
        float cR;
        float cI;
        if constexpr (TRANS == CtpsvTrans::OP_C) { // conj(A[i,i−1])·x_i
            cR = aTR * xIR + aTI * xII;
            cI = aTR * xII - aTI * xIR;
        } else {
            cR = aTR * xIR - aTI * xII;
            cI = aTR * xII + aTI * xIR;
        }
        const float xMR = bmR - s1R - cR;
        const float xMI = bmI - s1I - cI;
        bAll.SetValue(i, xIR);
        bAll.SetValue(iPlane + i, xII);
        bAll.SetValue(i - 1, xMR);
        bAll.SetValue(iPlane + i - 1, xMI);
        SetFlag<HardEvent::S_V>(CTPSV_EVT_S_V); // S 写 x_i/x_{i−1} → 下对 MAC 可见
        WaitFlag<HardEvent::S_V>(CTPSV_EVT_S_V);
    }

    __aicore__ inline void SolveRowDotBlk2()
    {
        // 杆 1 同域入口预清零（tR/tI 尾垫 + w 二级读域恒零垫；lenA 逐对非降）
        PreZeroDotBuffers();
        if (n <= 1) {
            PipeBarrier<PIPE_ALL>();
            return; // x_0 = b_0（UNIT，空 dot，b 平面原值即解）
        }
        // 奇 n 首行 x_{n−1} = b_{n−1}（UNIT 空 dot，零写回）；行对自奇 i 起
        const int32_t iTop = (n & 1) ? (n - 2) : (n - 1);
        // prologue：首对双窗装载（PF1/PF2 独立 id 各一次 set）
        {
            const int32_t lb0 = iTop & ~(static_cast<int32_t>(CTPSV_FLOAT_ALIGN) - 1);
            DrainVBeforeColLoad();
            IssueWindow(iTop - 1, lb0, 0);
            SetFlag<HardEvent::MTE2_V>(CTPSV_EVT_MTE2_V_PF);
            IssueWindow(iTop, lb0, 1);
            SetFlag<HardEvent::MTE2_V>(CTPSV_EVT_MTE2_V_PF2);
        }
        for (int32_t i = iTop; i >= 1; i -= 2) {
            PipeBarrier<PIPE_V>(); // 上对向量写 → 本对标量读序（遗留同位）
            const int32_t lb = i & ~(static_cast<int32_t>(CTPSV_FLOAT_ALIGN) - 1);
            const int32_t maskLen = (i + 1) - lb; // 两窗同掩蔽域 [lb, i]
            float aTR;
            float aTI;
            Blk2DotWindow1(i, lb, aTR, aTI);
            DotWindowL1(lb, maskLen, 0); // s'_{i−1} 部分和（dot 段 [i+1, n)，尾项后置）
            Blk2DotWindow2(i, lb);
            DotWindowL1(lb, maskLen, 128); // s_i 部分和（dot 段 [i+1, n)）
            float s1R;
            float s1I;
            float s2R;
            float s2I;
            {
                LocalTensor<float> w = bufW.Get<float>();
                LocalTensor<float> w2 = w[CTPSV_W_SPAN];
                CascadeSum4(w, w2, s1R, s1I, s2R, s2I);
            }
            Blk2DotStorePair(i, aTR, aTI, s1R, s1I, s2R, s2I);
        }
        PipeBarrier<PIPE_ALL>();
    }

    // 标量朴素复数除法 p/d（OP_C：p/conj(d) = p·d/|d|² → invIm 符号位编译期翻转；
    // probe_complex_div 定谳锁定形态：denom=dr²+di²，invRe/invIm + 2 次乘减合成）
    __aicore__ inline void ComplexDivide(float pR, float pI, float dr, float di, float& xRe, float& xIm) const
    {
        const float denom = dr * dr + di * di;
        const float invRe = dr / denom;
        // OP_C：x = p·conj(conj(d))/|d|² = p·d/|d|² → invIm = +di/denom（编译期翻转）；
        // OP_N/OP_T：x = p·conj(d)/|d|² → invIm = -di/denom
        const float invIm = (TRANS == CtpsvTrans::OP_C) ? (di / denom) : (-di / denom);
        xRe = pR * invRe - pI * invIm;
        xIm = pR * invIm + pI * invRe;
    }

    // 对角元独立 8B 微拷贝提取（bufDiag 头部；遗留路径：T1 全步 / 搭车时序不可达与主段
    // 空尾步，ctrsv LoadDiagScalar 范式）：LOWER 对角 = 列 j 首元素 / UPPER 对角 = 列 j 末元素
    __aicore__ inline void LoadDiagScalar(int32_t j, float& dr, float& di)
    {
        LocalTensor<float> d = bufDiag.Get<float>();
        const int64_t gmOff = 2LL * ColElemOff(j, j);
        DataCopyExtParams cp{1, 2U * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        DataCopyPad(d, apGM[gmOff], cp, pad);
        SetFlag<HardEvent::MTE2_V>(CTPSV_EVT_MTE2_V);
        WaitFlag<HardEvent::MTE2_V>(CTPSV_EVT_MTE2_V);
        dr = d.GetValue(0);
        di = d.GetValue(1);
    }

    // opt2 P0：搭车对角标量直读（col slot 内 elemIdx 元素处的 8B 交错对；调用前须已
    // WaitColChunk——MTE2_V_PF wait 覆盖标量读，LoadX STRIDED / stpsv ComputeMainFast
    // d=buf.GetValue(dot) 同构先例）
    __aicore__ inline void ReadDiagFromCol(const LocalTensor<float>& colIl, int32_t elemIdx)
    {
        diagRe = colIl.GetValue(2 * elemIdx);
        diagIm = colIl.GetValue(2 * elemIdx + 1);
    }

    // V1 头段 [s0, r0) 标量复 MAC 核心（源张量 + 源元素基 elem0——bufDiag 微拷贝与
    // opt2 P0 列段搭车两用；b_r -= (ar·xRe − ai·xIm) + i(ar·xIm + ai·xRe)，≤7 元素，
    // ctrsv ApplyHeadMac 范式；算术语序与迭代三逐位一致）
    __aicore__ inline void HeadMacCore(
        const LocalTensor<float>& src, int32_t elem0, int32_t s0, int32_t r0, float xRe, float xIm)
    {
        LocalTensor<float> bAll = bufB.Get<float>();
        for (int32_t k = 0; k < r0 - s0; k++) {
            const int32_t r = s0 + k;
            const float ar = src.GetValue(2 * (elem0 + k));
            const float ai = src.GetValue(2 * (elem0 + k) + 1);
            const float br = bAll.GetValue(r);
            const float bi = bAll.GetValue(iPlane + r);
            bAll.SetValue(r, br - (ar * xRe - ai * xIm));
            bAll.SetValue(iPlane + r, bi - (ar * xIm + ai * xRe));
        }
        PipeBarrier<PIPE_V>();
    }

    // 头段微拷贝装载（遗留路径共用）：AP 列 col 行段 [s0, r0) → bufDiag，MTE2→V
    // 同步后返回头段张量视图（hLen ≤ 8 元素，bufDiag 16 float 容量内）
    __aicore__ inline LocalTensor<float> LoadHeadSeg(int32_t col, int32_t s0, int32_t r0)
    {
        const int32_t hLen = r0 - s0;
        LocalTensor<float> head = bufDiag.Get<float>();
        const int64_t gmOff = 2LL * ColElemOff(col, s0);
        DataCopyExtParams cp{1, 8U * static_cast<uint32_t>(hLen), 0, 0, 0};
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        DataCopyPad(head, apGM[gmOff], cp, pad);
        SetFlag<HardEvent::MTE2_V>(CTPSV_EVT_MTE2_V);
        WaitFlag<HardEvent::MTE2_V>(CTPSV_EVT_MTE2_V);
        return head;
    }

    // V1 头段遗留路径（主段空尾步）：微拷贝 bufDiag → 核心（迭代三形态原样）
    __aicore__ inline void ApplyHeadMac(int32_t col, int32_t s0, int32_t r0, float xRe, float xIm)
    {
        if (r0 <= s0) {
            return;
        }
        HeadMacCore(LoadHeadSeg(col, s0, r0), 0, s0, r0, xRe, xIm);
    }

    // opt4 L3：V1 头段搭车标量路径（opt2 P0 ApplyHeadMacCol）移除——merged 步头段
    //   并入 chunk0 宽向量 Axpy（ConsumeColChunkMerged 前缀掩蔽），仅遗留尾步路径存续

    // V2 头段 [s0, r0) 标量复 dot 核心（源张量 + 源元素基；s += conj(a)·b（OP_C）/ a·b
    // （OP_T），≤7 元素，ctrsv ApplyHeadDot 范式；算术语序与迭代三逐位一致——头段先于
    // chunk 部分和累加进 sR/sI）
    __aicore__ inline void HeadDotCore(
        const LocalTensor<float>& src, int32_t elem0, int32_t s0, int32_t r0, float& sR, float& sI)
    {
        LocalTensor<float> bAll = bufB.Get<float>();
        for (int32_t k = 0; k < r0 - s0; k++) {
            const int32_t r = s0 + k;
            const float ar = src.GetValue(2 * (elem0 + k));
            const float ai = src.GetValue(2 * (elem0 + k) + 1);
            const float br = bAll.GetValue(r);
            const float bi = bAll.GetValue(iPlane + r);
            if constexpr (TRANS == CtpsvTrans::OP_C) {
                sR += ar * br + ai * bi; // conj(a)·b 实部
                sI += ar * bi - ai * br; // conj(a)·b 虚部
            } else {
                sR += ar * br - ai * bi;
                sI += ar * bi + ai * br;
            }
        }
        PipeBarrier<PIPE_V>();
    }

    // V2 头段遗留路径（主段空尾步）：微拷贝 bufDiag → 核心（迭代三形态原样）
    __aicore__ inline void ApplyHeadDot(int32_t col, int32_t s0, int32_t r0, float& sR, float& sI)
    {
        if (r0 <= s0) {
            return;
        }
        HeadDotCore(LoadHeadSeg(col, s0, r0), 0, s0, r0, sR, sI);
    }

    // opt4 L3：V2 头段搭车标量路径（opt2 P0 ApplyHeadDotCol）移除——merged 步头段
    //   并入 chunk0 宽向量 dot（DotChunk t 掩蔽），仅遗留尾步路径存续

    // V1 trailing 更新：b[base..] −= aCol·x_j 复 MAC 双平面 4 条宽 Axpy
    //   bR −= aR·xRe − aI·xIm；bI −= aR·xIm + aI·xRe
    // （计数区只包纯 Axpy：SetMaskCount 于区界 + AxpySeg 内 SetVectorMask 显式成对，
    //  即时 SetMaskNorm/ResetMask 收口，ctrsv TrailingAxpy 范式）
    __aicore__ inline void TrailingAxpy(int32_t base, int32_t len, float xRe, float xIm, bool exactCount = false)
    {
        LocalTensor<float> bAll = bufB.Get<float>();
        LocalTensor<float> aR = bufAR.Get<float>();
        LocalTensor<float> aI = bufAI.Get<float>();
        SetMaskCount();
        AxpySeg(bAll[base], aR, -xRe, len, exactCount);          // bR -= aR·xRe
        AxpySeg(bAll[base], aI, xIm, len, exactCount);           // bR += aI·xIm
        AxpySeg(bAll[iPlane + base], aR, -xIm, len, exactCount); // bI -= aR·xIm
        AxpySeg(bAll[iPlane + base], aI, -xRe, len, exactCount); // bI -= aI·xRe
        SetMaskNorm();
        ResetMask();
        PipeBarrier<PIPE_V>();
    }

    // Counter mask 宽向量 Axpy（SetMaskCount 计数区内显式 SetVectorMask 成对；ceil8
    // 补齐计数——pad lane × a_pad(0) 零贡献；repeatTimes uint8 ≤ 255：chunk ≤ 2048
    // → rep ≤ 32 单发；同宽 4 条共设计数区，ctrsv AxpySeg 范式）
    __aicore__ inline void AxpySeg(
        const LocalTensor<float>& dst, const LocalTensor<float>& src, float alpha, int32_t len, bool exactCount = false)
    {
        if (len <= 0) {
            return;
        }
        // exactCount（fl142_fix 20260930）：计数取精确 len——pad lane [len, ceil8(len))
        // 不写。COUNTER 掩码支持任意元素计数（CANN Tanh/Cos 内部实现
        // SetVectorMask<COUNTER>(0, tail=calCount%stackSize) 先例）；rep 仍按
        // ceil8(len) 覆盖（末 repeat 部分掩蔽）。默认 false = 现行 ceil8 计数，
        // 有限路径逐位不变（986 已过 case 零扰动）。
        const int32_t lenA = exactCount ? len : CtpsvCeil8(len);
        SetVectorMask<float, MaskMode::COUNTER>(lenA);
        const uint8_t rep = static_cast<uint8_t>((CtpsvCeil8(len) + 63) / 64);
        Axpy<float, float, false>(dst, src, alpha, MASK_PLACEHOLDER, rep, {1, 1, 8, 8});
    }

    // V2 复 dot 主段：tR/tI = Σ a·b 逐元素部分和（count 形式 Mul×4 + Sub/Add×2 自含掩码，
    // ctrsv DotChunk 范式）→ RRS 级联全和（WholeReduceSum 禁用于 count>64 全和，
    // DESIGN Contract）→ 标量累加，全程 UB 内。
    //   OP_T：tR = aR·bR − aI·bI；tI = aR·bI + aI·bR
    //   OP_C（迭代三符号模式融合，probe_opc_conj 位级一致实证）：conj(a)·b 展开——
    //     tR = aR·bR + aI·bI；tI = aR·bI − aI·bR（aI 平面不再 Muls(-1) 预翻转）
    __aicore__ inline void DotChunk(int32_t base, int32_t len, int32_t maskLen, float& sR, float& sI, bool monoZero)
    {
        LocalTensor<float> bAll = bufB.Get<float>();
        LocalTensor<float> aR = bufAR.Get<float>(), aI = bufAI.Get<float>();
        LocalTensor<float> tR = bufTR.Get<float>(), tI = bufTI.Get<float>(), t2 = bufT2.Get<float>();
        LocalTensor<float> w = bufW.Get<float>(), w2 = w[CTPSV_W_SPAN];
        const int32_t lenA = CtpsvCeil8(len);
        const LocalTensor<float> bR = bAll[base];
        const LocalTensor<float> bI = bAll[iPlane + base];
        Mul(tR, aR, bR, static_cast<uint32_t>(lenA));
        PipeBarrier<PIPE_V>();
        Mul(t2, aI, bI, static_cast<uint32_t>(lenA));
        PipeBarrier<PIPE_V>();
        if constexpr (TRANS == CtpsvTrans::OP_C) {
            Add(tR, tR, t2, static_cast<uint32_t>(lenA)); // tR = aR·bR + aI·bI（conj(a)·b 实部）
        } else {
            Sub(tR, tR, t2, static_cast<uint32_t>(lenA)); // tR = aR·bR − aI·bI
        }
        PipeBarrier<PIPE_V>();
        Mul(tI, aR, bI, static_cast<uint32_t>(lenA));
        PipeBarrier<PIPE_V>();
        Mul(t2, aI, bR, static_cast<uint32_t>(lenA));
        PipeBarrier<PIPE_V>();
        if constexpr (TRANS == CtpsvTrans::OP_C) {
            Sub(tI, tI, t2, static_cast<uint32_t>(lenA)); // tI = aR·bI − aI·bR（conj(a)·b 虚部）
        } else {
            Add(tI, tI, t2, static_cast<uint32_t>(lenA)); // tI = aR·bI + aI·bR
        }
        PipeBarrier<PIPE_V>();
        // ⚠ 级联尾 lane 显式清零（迭代二真机实证 #1：dot 模式 n≥8 全败根因）：
        //   count 形式 Mul/Add/Sub 只写 [0, lenA)，RRS 一级按整 repeat 界读 ceil64(lenA)
        //   lane——尾 lane 为上一步残留/跨 launch 未初始化 UB，直接污染全和
        //  （stpsv ComputeChunk 用 raw Mul 整 repeat 界 + 双侧零垫规避；本实现保留
        //   count Mul，尾 lane 显式置零等效）。lenR ≤ align64(chunkPad) 恒在缓冲域内。
        //   opt2 P1（2026-09-26）：零垫执行形态由标量 SetValue 环（[lenA,lenR)×2 平面
        //   ~56 次标量 UB 写/步，msprof 画像 4~7% 墙钟）向量化为单条 Duplicate/平面
        //  （stpsv IssueMainFast pad 零垫收缩同域先例：窄域 ≤56 lane、count 形式非
        //   64 倍数、dst 子偏移 8-float 对齐——同架构真机交付形态）。位级等价：写域
        //   [lenA,lenR) 与写值 0.0f 均不变（lenA 8 倍数 → 子偏移 32B 对齐；lenR-lenA
        //   ∈ [8,56] 8 倍数单 repeat）；tR/tI 为 V 专属缓冲（MTE2 仅写 col slot），
        //   与在飞预取装载区不相交，无涉 V_MTE2 事件对；RRS 护栏消费域（迭代二
        //   缺陷 #1 敏感区）逐位保持——多 chunk bit-exact 回归锚封闭。
        const int32_t lenR = CtpsvCeil64(lenA);
        // opt4 L3 头段窗口化：垃圾+对角 lane [0,maskLen) 后置掩蔽（Mul 链后、RRS 前）
        //   ——a 垃圾 lane（前列尾数据）× 未解 b lane 可能产 Inf/NaN，前置 a 清零仍有
        //   0×Inf=NaN 暴露（乘法不清 NaN），故必须乘加合成后掩 t（文件头 opt2 P0
        //   「无 0·Inf 暴露面」设计点的 L3 承载形式）；掩蔽写域 [0,maskLen) 与
        //   [lenA,lenR) 尾垫不相交（maskLen ≤ 8 ≤ lenA）
        // opt5 杆 1：monoZero（整列单块）档尾垫 [lenA,lenR) 免每步清零——lenA 逐步
        //   非降（LOWER len'=n-lb / UPPER len=i，见 SolveRowDot 谓词注释），尾垫 lane
        //   从未被 Mul 链写过 → SolveRowDot 入口预清零恒为零垫，位级等价；掩蔽 barrier
        //   随两组 Duplicate 同步收缩（全免时 tI 合成尾 PipeBarrier 已序 RRS 读）。
        const bool tailZero = (lenR > lenA) && !monoZero;
        if (maskLen > 0) {
            Duplicate(tR, 0.0f, static_cast<uint32_t>(maskLen));
            Duplicate(tI, 0.0f, static_cast<uint32_t>(maskLen));
        }
        if (tailZero) {
            Duplicate(tR[lenA], 0.0f, static_cast<uint32_t>(lenR - lenA));
            Duplicate(tI[lenA], 0.0f, static_cast<uint32_t>(lenR - lenA));
        }
        if (maskLen > 0 || tailZero) {
            PipeBarrier<PIPE_V>();
        }
        // opt4 L4：级联融合单趟化（sR/sI 双平面一次收束，位级等价见 CtpsvCascadeSum2）
        float cR = 0.0f, cI = 0.0f;
        CtpsvCascadeSum2(tR, tI, w, w2, lenA, cR, cI, monoZero);
        sR += cR;
        sI += cI;
    }

    // ── Stage 2：解写回 UB→GM x 原地址覆写 ──
    //  FAST：SoA→AoS 交织（Gather 256 分片；偏移相对各分片源基址 bAll[c/2]，分片无关）
    //        → DataCopyPad 整段写出（3 参形态）；
    //  GATHER：写回源 = 单一 concat 张量 [bR | bI | 跨度装载像]（bufB 自身）——gap 槽位
    //        偏移回指装载像原值（同值覆写 = gap 保持，probe_deinterleave/ctrsv 机制镜像），
    //        逻辑槽位 p=q·|incx|（xNeg 时 i=n-1-q）回指 bR[i]/bI[i]；分片（WB_CHUNK）
    //        Gather 散回暂存 + DataCopyPad 整段回写 GM；
    //  STRIDED：逐元素 strided 微拷贝（gap 物理零触碰）。
    // GATHER 写回：写回源 = 单一 concat 张量 [bR | bI | 跨度装载像]（bufB 自身）——
    //   gap 槽位偏移回指装载像原值（同值覆写 = gap 保持，probe_deinterleave/ctrsv 机制
    //   镜像），逻辑槽位 p=q·|incx|（xNeg 时 i=n-1-q）回指 bR[i]/bI[i]；分片（WB_CHUNK）
    //   Gather 散回暂存 + DataCopyPad 整段回写 GM
    __aicore__ inline void WriteBackGather(LocalTensor<float> bAll)
    {
        LocalTensor<float> outIl = bufXTrans.Get<float>();
        LocalTensor<int32_t> offWb = bufOffWb.Get<int32_t>();
        const int32_t imgBase = 2 * nPad; // 装载像在 bufB 内的 float 基址
        for (int32_t pc = 0; pc < spanI; pc += static_cast<int32_t>(CTPSV_WB_CHUNK)) {
            const int32_t len = (static_cast<int32_t>(CTPSV_WB_CHUNK) < spanI - pc) ?
                                    static_cast<int32_t>(CTPSV_WB_CHUNK) :
                                    (spanI - pc);
            // 仿射初值：全体槽位默认 gap → 装载像 (imgBase + 2·pc + k)·4B（全宽建表，
            // Gather 计数 2·len 只消费前 2·len 项）
            CtpsvBuildAffineOffsets(
                offWb, (imgBase + 2 * pc) * static_cast<int32_t>(sizeof(float)), static_cast<int32_t>(sizeof(float)),
                2 * static_cast<int32_t>(CTPSV_WB_CHUNK));
            PipeBarrier<PIPE_V>();
            // 逻辑槽位定点覆写：p = q·|incx| ∈ [pc, pc+len) → 实/虚源 bR[i]/bI[i]
            const int32_t q0 = (pc + incxAbs - 1) / incxAbs;
            for (int32_t q = q0; q * incxAbs < pc + len && q < n; ++q) {
                const int32_t p = q * incxAbs;
                const int32_t i = (xNeg != 0) ? (n - 1 - q) : q;
                offWb.SetValue(2 * (p - pc), i * static_cast<int32_t>(sizeof(float)));
                offWb.SetValue(2 * (p - pc) + 1, (iPlane + i) * static_cast<int32_t>(sizeof(float)));
            }
            PipeBarrier<PIPE_ALL>(); // S→V（表重建；亦覆盖解平面 S 写 → Gather 读）
            CtpsvMaskFull();         // Gather 须在 normal mask 区执行
            Gather(outIl, bAll, offWb.ReinterpretCast<uint32_t>(), 0, 2 * len);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(CTPSV_EVT_V_MTE3); // V 写 outIl → MTE3 读
            WaitFlag<HardEvent::V_MTE3>(CTPSV_EVT_V_MTE3);
            DataCopyExtParams cpW{1, 8U * static_cast<uint32_t>(len), 0, 0, 0};
            DataCopyPad(xGM[2 * pc], outIl, cpW);         // UB→GM 3 参形态
            SetFlag<HardEvent::MTE3_V>(CTPSV_EVT_MTE3_V); // 写回分片循环 outIl 复用防护
            WaitFlag<HardEvent::MTE3_V>(CTPSV_EVT_MTE3_V);
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void WriteBackX()
    {
        LocalTensor<float> bAll = bufB.Get<float>();
        if (path == CtpsvXPath::FAST) {
            LocalTensor<float> outIl = bufXTrans.Get<float>();
            LocalTensor<int32_t> off = bufOff.Get<int32_t>();
            LocalTensor<uint32_t> offInter = off.ReinterpretCast<uint32_t>();
            const int32_t outLen = CtpsvCeil8(2 * n);
            for (int32_t c = 0; c < outLen; c += static_cast<int32_t>(CTPSV_GATHER_CHUNK)) {
                const int32_t l = (outLen - c < static_cast<int32_t>(CTPSV_GATHER_CHUNK)) ?
                                      (outLen - c) :
                                      static_cast<int32_t>(CTPSV_GATHER_CHUNK);
                Gather(outIl[c], bAll[c / 2], offInter, 0, l); // normal mask 区执行
            }
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(CTPSV_EVT_V_MTE3); // V 写（Axpy/Mul trailing）→ MTE3 读
            WaitFlag<HardEvent::V_MTE3>(CTPSV_EVT_V_MTE3);
            SetFlag<HardEvent::S_MTE3>(CTPSV_EVT_S_MTE3); // S 写（对角 SetValue）→ MTE3 读
            WaitFlag<HardEvent::S_MTE3>(CTPSV_EVT_S_MTE3);
            DataCopyExtParams cp{1, 8U * static_cast<uint32_t>(n), 0, 0, 0};
            DataCopyPad(xGM, outIl, cp); // UB→GM 3 参形态（本 CANN 无 pad 参数重载）
            PipeBarrier<PIPE_ALL>();
        } else if (path == CtpsvXPath::GATHER) {
            WriteBackGather(bAll);
        } else {
            LocalTensor<float> el = bufEl.Get<float>();
            DataCopyExtParams cp{1, 8U, 0, 0, 0};
            for (int32_t i = 0; i < n; ++i) {
                el.SetValue(0, bAll.GetValue(i));
                el.SetValue(1, bAll.GetValue(iPlane + i));
                SetFlag<HardEvent::S_MTE3>(CTPSV_EVT_S_MTE3); // S 写 el → MTE3 读
                WaitFlag<HardEvent::S_MTE3>(CTPSV_EVT_S_MTE3);
                DataCopyPad(xGM[2 * Phys(i)], el, cp);        // gap 槽位从不触碰 → 原样保留
                SetFlag<HardEvent::MTE3_V>(CTPSV_EVT_MTE3_V); // el 复用防护
                WaitFlag<HardEvent::MTE3_V>(CTPSV_EVT_MTE3_V);
            }
            PipeBarrier<PIPE_ALL>();
        }
    }

    TPipe pipe;
    TBuf<TPosition::VECCALC> bufB;       // bR|bI 双平面常驻（+GATHER 跨度装载像尾接）
    TBuf<TPosition::VECCALC> bufXTrans;  // FAST: 装载 vred 源 / 写回交织暂存；GATHER: 写回分片暂存
    TBuf<TPosition::VECCALC> bufCol;     // AP 列段 AoS 交错装载（vred 源；dbuf 双 slot 轮转）
    TBuf<TPosition::VECCALC> bufAR;      // 列段解交织实平面（vred 目的）
    TBuf<TPosition::VECCALC> bufAI;      // 列段解交织虚平面（vred 目的；OP_C 符号融合于 dot 承载）
    TBuf<TPosition::VECCALC> bufTR;      // dot 复乘实部部分和（仅 V2）
    TBuf<TPosition::VECCALC> bufTI;      // dot 复乘虚部部分和（仅 V2）
    TBuf<TPosition::VECCALC> bufT2;      // dot 复乘 scratch（仅 V2）
    TBuf<TPosition::VECCALC> bufW;       // dot RRS 级联 w/w2（2×256，dst 子偏移 0/256 均 32B 对齐）
    TBuf<TPosition::VECCALC> bufOffElem; // GATHER 逻辑 i→装载像实部字节偏移表
    TBuf<TPosition::VECCALC> bufOffOdd;  // GATHER 逻辑 i→装载像虚部字节偏移表
    TBuf<TPosition::VECCALC> bufOffWb;   // GATHER 写回分片逐 float 偏移表
    TBuf<TPosition::VECCALC> bufEl;      // STRIDED 单元素微拷贝
    TBuf<TPosition::VECCALC> bufDiag;    // 对角元 8B 微拷贝 + 头段微拷贝（迭代三移驻）
    TBuf<TPosition::VECCALC> bufOff;     // FAST 写回交织偏移表（256 int32，Init 一次复用）
    GlobalTensor<float> apGM;
    GlobalTensor<float> xGM;
    int32_t blockIdx = 0;
    int32_t blockNum = 1;
    int32_t n = 0;
    int32_t incx = 1;
    int32_t incxAbs = 1;
    int32_t xNeg = 0;
    int32_t chunkLen = 1;
    int32_t nPad = 64;
    int32_t iPlane = 64;
    int32_t chunkPad = 8;
    int32_t xLen8 = 8;
    int32_t spanI = 0;  // 仅 GATHER 有效（护栏内必有界）
    int32_t span8I = 0; // 仅 GATHER 有效（装载像对齐宽度）
    int64_t span64 = 1;
    int32_t colFloats = 128; // col 单 slot float 宽度（align64(2·chunkPad)）
    int32_t planeRide = 0;   // opt4 L3：LOWER 头段窗口化 plane 增量（chunkLen<n 多 chunk 档
                             //   = 64：aR/aI（V1）/tR/tI/t2（V2）需容纳 ceil64(chunkPad+8)；
                             //   整列单块档 chunkLen==n = 0，UB 布局与 opt3 终态逐字节一致）
    bool dbuf = false;       // 列段双 slot 预取乒乓使能（UB 预算内）
    int32_t colSlot = 0;     // 下一次 Issue 的 slot（全局轮转，含跨步 look-ahead）
    int32_t pfSlot = 0;      // 在飞预取的 slot
    float diagRe = 1.0f;     // opt2 P0：NON_UNIT 对角暂存（搭车 col slot 直读 / 遗留微拷贝填充，
    float diagIm = 0.0f;     //   ComplexDivide 消费；UNIT 不触碰）
    CtpsvXPath path = CtpsvXPath::FAST;

    // 步骤迭代方向（V1：LOWER 前代升序 / UPPER 回代降序；V2：UPPER 前代升序 / LOWER 回代降序）
    static constexpr bool kFwd = (TRANS == CtpsvTrans::OP_N) ? (UPLO == CtpsvUplo::LOWER) : (UPLO == CtpsvUplo::UPPER);
    // opt2 P0：搭车装载 col slot 增量（float）。LOWER（前缀 ≤8 元素/16 float + vred src
    // 偏移 16 float 后的 64-float 对齐越读护栏）与 UPPER V2 NON_UNIT（末块对角 +1 元素）
    // = 64；T1/T2/T4/T6 无搭车 = 0（UB 布局与迭代三一致，见 Init 注释）。
    static constexpr int32_t kColRideFloats =
        ((UPLO == CtpsvUplo::LOWER) || (TRANS != CtpsvTrans::OP_N && DIAG == CtpsvDiag::NON_UNIT)) ? 64 : 0;
};

// ==========================================================================
//  opt2 P2：小 n（≤8）FAST 标量快路径 —— 独立自由模板函数（noinline）
//  结构隔离纪律（P0 v6「T2 col 布局漂移」同款教训的代码布局变体，实测定谳）：
//  快路径若内联进 CtpsvKernel（入口函数含 12 模板热循环全量内联体），其代码体
//  移位 n>8 未改模板热循环布局——T7/T8 档 kernel-only +3.65%（scalar，msprof
//  TC_PF_1111 n=2309 定谳 1422.8→1474.8µs）。故：
//    ① CtpsvKernel 类体逐位保持 P1 形态（本快路径零触碰）；
//    ② 快路径 = 独立自由模板函数（自含 TPipe/TBuf，足迹独立于主类 UB 规划），
//      内联进 _small 分离入口（noinline 自由函数形态三轮实测静默产出错误结果，
//      见 CtpsvSmallSolve 头「内联纪律」）；
//    ③ 分发点 = host 侧 aclblasCtpsv（n/incx 已知，谓词求值后直调
//      ctpsv_kernel_do_small → _small 独立 kernel 符号）——主入口宏与主分发
//      逐位保持 P1，n>8 主路径 codegen 零扰动。
//  语义：x 与 packed AP 各一次 DataCopyPad 整段入 UB（AP ≤36 复数元素 288B），
//  纯标量求解（Netlib 镜像语序 = CPU golden 方言），一次 DataCopyPad 写回；
//  事件对 3 组；guard（incx==1 && n≤CTPSV_SMALL_N_MAX）在 host 侧 aclblasCtpsv
//  求值后经 ctpsv_kernel_do_small 直调 _small 独立符号（覆盖全部 12 模板，
//  V1 axpy / V2 dot(OP_T/OP_C 符号融合) / UNIT 直通 constexpr 全分支）。
// ==========================================================================

// packed 列内元素偏移（复数元素单位，快路径独立版——主类 ColElemOff 同式）
template <CtpsvUplo UPLO>
__aicore__ inline int32_t CtpsvSmallColOff(int32_t col, int32_t row, int32_t n)
{
    if constexpr (UPLO == CtpsvUplo::LOWER) {
        return static_cast<int32_t>(CtpsvLowerColStart(col, n)) + (row - col);
    } else {
        return static_cast<int32_t>(CtpsvUpperColStart(col)) + row;
    }
}

// 标量朴素复除（主类 ComplexDivide 同式自由版）
template <CtpsvTrans TRANS>
__aicore__ inline void CtpsvSmallDivide(float pR, float pI, float dr, float di, float& xRe, float& xIm)
{
    const float denom = dr * dr + di * di;
    const float invRe = dr / denom;
    const float invIm = (TRANS == CtpsvTrans::OP_C) ? (di / denom) : (-di / denom);
    xRe = pR * invRe - pI * invIm;
    xIm = pR * invIm + pI * invRe;
}

// 小 n（≤8）输入装载：x/AP 交错像 GM→UB 两段拷入，MTE2 按序放行标量读。
// xIl/apIl 由本函数回填（调用方寄存器驻留数组的读源/写宿）
__aicore__ inline void CtpsvSmallLoadIn(
    TPipe& pipe, TBuf<TPosition::VECCALC>& bufX, TBuf<TPosition::VECCALC>& bufAp, GM_ADDR apGm, GM_ADDR xGm, int32_t n,
    LocalTensor<float>& xIl, LocalTensor<float>& apIl)
{
    pipe.InitBuffer(bufX, static_cast<uint32_t>(CtpsvCeil64(2 * n)) * sizeof(float));
    pipe.InitBuffer(bufAp, static_cast<uint32_t>(CtpsvCeil64(n * (n + 1))) * sizeof(float));
    GlobalTensor<float> apGM;
    GlobalTensor<float> xGM;
    const uint64_t apCount = static_cast<uint64_t>(n) * (static_cast<uint64_t>(n) + 1) / 2;
    apGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(apGm), static_cast<uint32_t>(2ULL * apCount));
    xGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(xGm), static_cast<uint32_t>(2ULL * n));
    xIl = bufX.Get<float>();
    apIl = bufAp.Get<float>();
    DataCopyExtParams cx{1, 8U * static_cast<uint32_t>(n), 0, 0, 0};
    DataCopyPadExtParams<float> pad{false, 0, 0, 0};
    DataCopyPad(xIl, xGM, cx, pad); // x 整段（n 复数，incx=1 连续）
    DataCopyExtParams ca{1, 8U * static_cast<uint32_t>(apCount), 0, 0, 0};
    DataCopyPad(apIl, apGM, ca, pad); // packed AP 整段
    SetFlag<HardEvent::MTE2_V>(CTPSV_EVT_MTE2_V);
    WaitFlag<HardEvent::MTE2_V>(CTPSV_EVT_MTE2_V); // MTE2 按序：两拷贝均完成后放行标量读
}

// 小 n 对角复除出 x_j（NON_UNIT：读 packed 对角元素后 CtpsvSmallDivide；UNIT 直通
// 不读对角、无除法——spec diag 语义，与主路径一致）
template <CtpsvUplo UPLO, CtpsvTrans TRANS, CtpsvDiag DIAG>
__aicore__ inline void CtpsvSmallSolveX(
    const float pR, const float pI, const LocalTensor<float>& apIl, const int32_t j, const int32_t n, float& xRe,
    float& xIm)
{
    if constexpr (DIAG == CtpsvDiag::NON_UNIT) {
        const int32_t dOff = 2 * CtpsvSmallColOff<UPLO>(j, j, n);
        CtpsvSmallDivide<TRANS>(pR, pI, apIl.GetValue(dOff), apIl.GetValue(dOff + 1), xRe, xIm);
    } else {
        xRe = pR;
        xIm = pI;
    }
}

// 小 n 复 dot 标量累加：s += a·b（OP_C 为 conj(a)·b，符号位编译期翻转；
// 累加语序与主类 HeadDotCore 同式）
template <CtpsvTrans TRANS>
__aicore__ inline void CtpsvSmallDotAccum(
    const float ar, const float ai, const float br, const float bi, float& sR, float& sI)
{
    if constexpr (TRANS == CtpsvTrans::OP_C) {
        sR += ar * br + ai * bi;
        sI += ar * bi - ai * br;
    } else {
        sR += ar * br - ai * bi;
        sI += ar * bi + ai * br;
    }
}

// 小 n V1 trailing 行区间（LOWER (j,n) / UPPER [0,j)；编译期分支）
template <CtpsvUplo UPLO>
__aicore__ inline void CtpsvSmallTrailingRange(const int32_t j, const int32_t n, int32_t& r0, int32_t& r1)
{
    if constexpr (UPLO == CtpsvUplo::LOWER) {
        r0 = j + 1;
        r1 = n;
    } else {
        r0 = 0;
        r1 = j;
    }
}

// 小 n V2 dot 行区间（UPPER [0,i) / LOWER (i,n)；编译期分支）
template <CtpsvUplo UPLO>
__aicore__ inline void CtpsvSmallDotRange(const int32_t i, const int32_t n, int32_t& r0, int32_t& r1)
{
    if constexpr (UPLO == CtpsvUplo::UPPER) {
        r0 = 0;
        r1 = i;
    } else {
        r0 = i + 1;
        r1 = n;
    }
}

// 小 n 复 trailing MAC：b_r ← b_r − a·x（主类 HeadMacCore 同式；实部/虚部两条独立
// 计算，读取先于写回）
__aicore__ inline void CtpsvSmallTrailingMac(
    const float ar, const float ai, const float xRe, const float xIm, float& bR, float& bI)
{
    bR = bR - (ar * xRe - ai * xIm);
    bI = bI - (ar * xIm + ai * xRe);
}

// 小 n 尾段写出：x 寄存器像已写回 xIl（UB），S→MTE3 事件对后单段拷出 GM
__aicore__ inline void CtpsvSmallStoreOut(const LocalTensor<float>& xIl, GM_ADDR xGm, const int32_t n)
{
    GlobalTensor<float> xGM;
    xGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(xGm), static_cast<uint32_t>(2ULL * n));
    DataCopyExtParams cx{1, 8U * static_cast<uint32_t>(n), 0, 0, 0};
    SetFlag<HardEvent::S_MTE3>(CTPSV_EVT_S_MTE3); // S 写 xIl → MTE3 读
    WaitFlag<HardEvent::S_MTE3>(CTPSV_EVT_S_MTE3);
    DataCopyPad(xGM, xIl, cx); // UB→GM 3 参形态（本 CANN 无 pad 参数重载）
    PipeBarrier<PIPE_ALL>();
}

// ⚠ 内联纪律（真机实证，opt2 P2 三轮二分定谳）：本函数必须内联进 __global__
//   __vector__ 入口（默认 inline，禁 noinline）——noinline 自由函数形态在 v2/v3/
//   v3b 三轮实测中静默产出错误结果（n=1 全错、小 n 段批量 FAIL，acc 二分
//   定谳），与文件头「__vector__ 属性为向量特性硬约束（ctrsv probe 实证）」同源
//   ——脱离 __vector__ 编译上下文的 kernel 侧计算体不可信。_small 分离入口符号
//   已保证主路径 codegen 零扰动（T7 档 n>8 逐位一致 1420.5µs 实证），无需 noinline。
template <CtpsvUplo UPLO, CtpsvTrans TRANS, CtpsvDiag DIAG>
__aicore__ inline void CtpsvSmallSolve(GM_ADDR apGm, GM_ADDR xGm, int32_t n)
{
    TPipe pipe;
    TBuf<TPosition::VECCALC> bufX;  // x 交错像（2n float，原位求解）
    TBuf<TPosition::VECCALC> bufAp; // packed AP 交错像（n(n+1) float）
    LocalTensor<float> xIl, apIl;
    CtpsvSmallLoadIn(pipe, bufX, bufAp, apGm, xGm, n, xIl, apIl);

    // opt3 E2：x 寄存器驻留——MTE2_V wait 后 x 一次读入局部数组，全程寄存器求解
    //   （每 MAC 4 读 2 写的 UB 往返消除），尾段一次写回 2n SetValue。MAC 语序逐位
    //   不变（仅存储位置变化，golden 同序纪律保持）。数组界 n≤CTPSV_SMALL_N_MAX 由
    //   host 分发谓词保证（CTPSV_SMALL_N_MAX 共享真源）。
    float xr[CTPSV_SMALL_N_MAX], xi[CTPSV_SMALL_N_MAX];
    for (int32_t i = 0; i < n; ++i) {
        xr[i] = xIl.GetValue(2 * i);
        xi[i] = xIl.GetValue(2 * i + 1);
    }
    if constexpr (TRANS == CtpsvTrans::OP_N) {
        // V1 右视列 axpy：LOWER 前代 j 升序 / UPPER 回代 j 降序（golden 同向）
        constexpr bool FORWARD = (UPLO == CtpsvUplo::LOWER);
        for (int32_t j = (FORWARD ? 0 : n - 1); (FORWARD ? j < n : j >= 0); (FORWARD ? ++j : --j)) {
            float xRe, xIm;
            CtpsvSmallSolveX<UPLO, TRANS, DIAG>(xr[j], xi[j], apIl, j, n, xRe, xIm);
            xr[j] = xRe;
            xi[j] = xIm;
            int32_t r0, r1; // trailing 行区间（LOWER (j,n) / UPPER [0,j)）
            CtpsvSmallTrailingRange<UPLO>(j, n, r0, r1);
            for (int32_t r = r0; r < r1; ++r) { // 逐 b_r 独立更新，r 向不影响结果（golden 同式）
                const int32_t o = 2 * CtpsvSmallColOff<UPLO>(j, r, n);
                CtpsvSmallTrailingMac(apIl.GetValue(o), apIl.GetValue(o + 1), xRe, xIm, xr[r], xi[r]);
            }
        }
    } else {
        // V2 左视行 dot：UPPER 前代 i 升序（dot 行 [0,i)，对角=列末）/ LOWER 回代 i 降序
        // （dot 行 (i,n)，对角=列首）；dot 累加序与 golden 逐位同向（UPPER 升/LOWER 降）
        constexpr bool FORWARD = (UPLO == CtpsvUplo::UPPER);
        for (int32_t i = (FORWARD ? 0 : n - 1); (FORWARD ? i < n : i >= 0); (FORWARD ? ++i : --i)) {
            float sR = 0.0f, sI = 0.0f;
            int32_t r0, r1; // dot 行区间（UPPER [0,i) / LOWER (i,n)）
            CtpsvSmallDotRange<UPLO>(i, n, r0, r1);
            for (int32_t r = r0; r < r1; ++r) {
                const int32_t o = 2 * CtpsvSmallColOff<UPLO>(i, r, n);
                CtpsvSmallDotAccum<TRANS>(apIl.GetValue(o), apIl.GetValue(o + 1), xr[r], xi[r], sR, sI);
            }
            float xRe, xIm;
            CtpsvSmallSolveX<UPLO, TRANS, DIAG>(xr[i] - sR, xi[i] - sI, apIl, i, n, xRe, xIm);
            xr[i] = xRe;
            xi[i] = xIm;
        }
    }

    // opt3 E2 尾段：x 寄存器像一次写回 UB（供 MTE3 写出 GM）
    for (int32_t i = 0; i < n; ++i) {
        xIl.SetValue(2 * i, xr[i]);
        xIl.SetValue(2 * i + 1, xi[i]);
    }
    CtpsvSmallStoreOut(xIl, xGm, n);
}

// ==========================================================================
//  Kernel entry points — 12 编译期模板全实例化（迭代二，DESIGN §3.2 表 A）
//  ⚠ __vector__ 属性为 vreducev2 硬约束（ctrsv probe_deinterleave 实证：无属性
//    ccec 报 target feature 错），12 模板统一加属性。
// ==========================================================================

#define CTPSV_KERNEL_ENTRY(U, T, D, SUFFIX)                                                              \
    __global__ __aicore__ __vector__ void ctpsv_kernel_##SUFFIX(GM_ADDR AP, GM_ADDR x, GM_ADDR tilingGm) \
    {                                                                                                    \
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);                                                  \
        CtpsvKernel<CtpsvUplo::U, CtpsvTrans::T, CtpsvDiag::D> op;                                       \
        op.Init(AP, x, tilingGm);                                                                        \
        op.Process();                                                                                    \
    }

// opt2 P2 小 n 标量快路径独立入口（分离符号结构隔离）：主入口宏（上方）逐位保持
// P1 形态——n>8 主路径 codegen 零扰动；_small 入口仅被 ctpsv_kernel_do_small
// （host 按 n≤CTPSV_SMALL_N_MAX && incx==1 直调）触达。guard 谓词在 host 侧
// aclblasCtpsv（n/incx 已知），kernel 侧无分支。
// opt3 E1（2026-09-27）：n 改作 launch 第 3 标量参数（host 已知 n，消 _small 入口
// tiling GM 依赖装载；arch35 stpsv/strsv TilingData 按值传 launch 官方先例，向上游
// 形态靠拢），tilingGm 参数整体移除——_small 全程零 tiling 读。⚠ v3b 教训边界澄清：
// 损坏形态 = noinline 自由函数经形参传递 GM_ADDR 派生值；本形态 = launch ABI 标量
// 参数 + CtpsvSmallSolve 内联消费，机制不同，A/B 实证见 opt3_e1_report.md。
#define CTPSV_SMALL_KERNEL_ENTRY(U, T, D, SUFFIX)                                                         \
    __global__ __aicore__ __vector__ void ctpsv_kernel_##SUFFIX##_small(GM_ADDR AP, GM_ADDR x, int32_t n) \
    {                                                                                                     \
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);                                                   \
        CtpsvSmallSolve<CtpsvUplo::U, CtpsvTrans::T, CtpsvDiag::D>(AP, x, n);                             \
    }

CTPSV_KERNEL_ENTRY(UPPER, OP_N, NON_UNIT, upper_no_trans_non_unit) // T1 回代 axpy
CTPSV_KERNEL_ENTRY(UPPER, OP_N, UNIT, upper_no_trans_unit)         // T2
CTPSV_KERNEL_ENTRY(UPPER, OP_T, NON_UNIT, upper_trans_non_unit)    // T3 前代 dot
CTPSV_KERNEL_ENTRY(UPPER, OP_T, UNIT, upper_trans_unit)            // T4
CTPSV_KERNEL_ENTRY(UPPER, OP_C, NON_UNIT, upper_conj_non_unit)     // T5 前代 dot（共轭独立分支）
CTPSV_KERNEL_ENTRY(UPPER, OP_C, UNIT, upper_conj_unit)             // T6
CTPSV_KERNEL_ENTRY(LOWER, OP_N, NON_UNIT, lower_no_trans_non_unit) // T7 前代 axpy（迭代一主路径）
CTPSV_KERNEL_ENTRY(LOWER, OP_N, UNIT, lower_no_trans_unit)         // T8
CTPSV_KERNEL_ENTRY(LOWER, OP_T, NON_UNIT, lower_trans_non_unit)    // T9 回代 dot
CTPSV_KERNEL_ENTRY(LOWER, OP_T, UNIT, lower_trans_unit)            // T10
CTPSV_KERNEL_ENTRY(LOWER, OP_C, NON_UNIT, lower_conj_non_unit)     // T11 回代 dot（共轭独立分支）
CTPSV_KERNEL_ENTRY(LOWER, OP_C, UNIT, lower_conj_unit)             // T12

// opt2 P2：小 n 标量快路径 12 模板独立实例（n≤8 且 incx=1，host 直调；分离符号
// 结构隔离见 CTPSV_SMALL_KERNEL_ENTRY 注释）
CTPSV_SMALL_KERNEL_ENTRY(UPPER, OP_N, NON_UNIT, upper_no_trans_non_unit)
CTPSV_SMALL_KERNEL_ENTRY(UPPER, OP_N, UNIT, upper_no_trans_unit)
CTPSV_SMALL_KERNEL_ENTRY(UPPER, OP_T, NON_UNIT, upper_trans_non_unit)
CTPSV_SMALL_KERNEL_ENTRY(UPPER, OP_T, UNIT, upper_trans_unit)
CTPSV_SMALL_KERNEL_ENTRY(UPPER, OP_C, NON_UNIT, upper_conj_non_unit)
CTPSV_SMALL_KERNEL_ENTRY(UPPER, OP_C, UNIT, upper_conj_unit)
CTPSV_SMALL_KERNEL_ENTRY(LOWER, OP_N, NON_UNIT, lower_no_trans_non_unit)
CTPSV_SMALL_KERNEL_ENTRY(LOWER, OP_N, UNIT, lower_no_trans_unit)
CTPSV_SMALL_KERNEL_ENTRY(LOWER, OP_T, NON_UNIT, lower_trans_non_unit)
CTPSV_SMALL_KERNEL_ENTRY(LOWER, OP_T, UNIT, lower_trans_unit)
CTPSV_SMALL_KERNEL_ENTRY(LOWER, OP_C, NON_UNIT, lower_conj_non_unit)
CTPSV_SMALL_KERNEL_ENTRY(LOWER, OP_C, UNIT, lower_conj_unit)

// ==========================================================================
//  Kernel dispatcher —— host 三层 switch 分发（uplo→trans→diag，ctrsv 12 分支范式）
//  复数纪律：OP_T（纯转置）与 OP_C（共轭转置）为不同算子，host/kernel 均不得折叠
//  （CP1 裁定第 7 条 / spec op_c_conjugate_transpose_identity）。
// ==========================================================================

// LOWER 半分发（uplo 已定，trans→diag 两层；T7/T8/T9/T10/T11/T12）
static void CtpsvKernelDoLower(
    uint8_t* AP, uint8_t* x, uint8_t* tilingGm, aclblasOperation_t trans, aclblasDiagType_t diag, uint32_t numBlocks,
    void* stream)
{
    if (trans == ACLBLAS_OP_N) {
        if (diag == ACLBLAS_NON_UNIT) {
            ctpsv_kernel_lower_no_trans_non_unit<<<numBlocks, nullptr, stream>>>(AP, x, tilingGm); // T7
        } else {
            ctpsv_kernel_lower_no_trans_unit<<<numBlocks, nullptr, stream>>>(AP, x, tilingGm); // T8
        }
    } else if (trans == ACLBLAS_OP_T) {
        if (diag == ACLBLAS_NON_UNIT) {
            ctpsv_kernel_lower_trans_non_unit<<<numBlocks, nullptr, stream>>>(AP, x, tilingGm); // T9
        } else {
            ctpsv_kernel_lower_trans_unit<<<numBlocks, nullptr, stream>>>(AP, x, tilingGm); // T10
        }
    } else {
        if (diag == ACLBLAS_NON_UNIT) {
            ctpsv_kernel_lower_conj_non_unit<<<numBlocks, nullptr, stream>>>(AP, x, tilingGm); // T11
        } else {
            ctpsv_kernel_lower_conj_unit<<<numBlocks, nullptr, stream>>>(AP, x, tilingGm); // T12
        }
    }
}

// UPPER 半分发（trans→diag 两层；T1/T2/T3/T4/T5/T6）
static void CtpsvKernelDoUpper(
    uint8_t* AP, uint8_t* x, uint8_t* tilingGm, aclblasOperation_t trans, aclblasDiagType_t diag, uint32_t numBlocks,
    void* stream)
{
    if (trans == ACLBLAS_OP_N) {
        if (diag == ACLBLAS_NON_UNIT) {
            ctpsv_kernel_upper_no_trans_non_unit<<<numBlocks, nullptr, stream>>>(AP, x, tilingGm); // T1
        } else {
            ctpsv_kernel_upper_no_trans_unit<<<numBlocks, nullptr, stream>>>(AP, x, tilingGm); // T2
        }
    } else if (trans == ACLBLAS_OP_T) {
        if (diag == ACLBLAS_NON_UNIT) {
            ctpsv_kernel_upper_trans_non_unit<<<numBlocks, nullptr, stream>>>(AP, x, tilingGm); // T3
        } else {
            ctpsv_kernel_upper_trans_unit<<<numBlocks, nullptr, stream>>>(AP, x, tilingGm); // T4
        }
    } else {
        if (diag == ACLBLAS_NON_UNIT) {
            ctpsv_kernel_upper_conj_non_unit<<<numBlocks, nullptr, stream>>>(AP, x, tilingGm); // T5
        } else {
            ctpsv_kernel_upper_conj_unit<<<numBlocks, nullptr, stream>>>(AP, x, tilingGm); // T6
        }
    }
}

void ctpsv_kernel_do(
    uint8_t* AP, uint8_t* x, uint8_t* tilingGm, aclblasFillMode_t uplo, aclblasOperation_t trans,
    aclblasDiagType_t diag, uint32_t numBlocks, void* stream)
{
    if (uplo == ACLBLAS_LOWER) {
        CtpsvKernelDoLower(AP, x, tilingGm, trans, diag, numBlocks, stream);
    } else {
        CtpsvKernelDoUpper(AP, x, tilingGm, trans, diag, numBlocks, stream);
    }
}

// ── opt2 P2：小 n（≤8）标量快路径分发（host 侧 guard：n≤CTPSV_SMALL_N_MAX 且
//    incx==1——谓词在 aclblasCtpsv 求值，UT 镜像同源；12 分支结构与主分发一致，
//    OP_C 独立分支不折叠纪律同守）。opt3 E1：n 作 launch 标量参数直传（签名第 3 参，
//    _small 全程零 tiling 读）。 ──
// 小 n 标量快路径 LOWER 半分发（trans→diag 两层；12 分支结构与主分发一致）
static void CtpsvKernelDoSmallLower(
    uint8_t* AP, uint8_t* x, int n, aclblasOperation_t trans, aclblasDiagType_t diag, uint32_t numBlocks, void* stream)
{
    if (trans == ACLBLAS_OP_N) {
        if (diag == ACLBLAS_NON_UNIT) {
            ctpsv_kernel_lower_no_trans_non_unit_small<<<numBlocks, nullptr, stream>>>(AP, x, static_cast<int32_t>(n));
        } else {
            ctpsv_kernel_lower_no_trans_unit_small<<<numBlocks, nullptr, stream>>>(AP, x, static_cast<int32_t>(n));
        }
    } else if (trans == ACLBLAS_OP_T) {
        if (diag == ACLBLAS_NON_UNIT) {
            ctpsv_kernel_lower_trans_non_unit_small<<<numBlocks, nullptr, stream>>>(AP, x, static_cast<int32_t>(n));
        } else {
            ctpsv_kernel_lower_trans_unit_small<<<numBlocks, nullptr, stream>>>(AP, x, static_cast<int32_t>(n));
        }
    } else {
        if (diag == ACLBLAS_NON_UNIT) {
            ctpsv_kernel_lower_conj_non_unit_small<<<numBlocks, nullptr, stream>>>(AP, x, static_cast<int32_t>(n));
        } else {
            ctpsv_kernel_lower_conj_unit_small<<<numBlocks, nullptr, stream>>>(AP, x, static_cast<int32_t>(n));
        }
    }
}

// 小 n 标量快路径 UPPER 半分发（trans→diag 两层）
static void CtpsvKernelDoSmallUpper(
    uint8_t* AP, uint8_t* x, int n, aclblasOperation_t trans, aclblasDiagType_t diag, uint32_t numBlocks, void* stream)
{
    if (trans == ACLBLAS_OP_N) {
        if (diag == ACLBLAS_NON_UNIT) {
            ctpsv_kernel_upper_no_trans_non_unit_small<<<numBlocks, nullptr, stream>>>(AP, x, static_cast<int32_t>(n));
        } else {
            ctpsv_kernel_upper_no_trans_unit_small<<<numBlocks, nullptr, stream>>>(AP, x, static_cast<int32_t>(n));
        }
    } else if (trans == ACLBLAS_OP_T) {
        if (diag == ACLBLAS_NON_UNIT) {
            ctpsv_kernel_upper_trans_non_unit_small<<<numBlocks, nullptr, stream>>>(AP, x, static_cast<int32_t>(n));
        } else {
            ctpsv_kernel_upper_trans_unit_small<<<numBlocks, nullptr, stream>>>(AP, x, static_cast<int32_t>(n));
        }
    } else {
        if (diag == ACLBLAS_NON_UNIT) {
            ctpsv_kernel_upper_conj_non_unit_small<<<numBlocks, nullptr, stream>>>(AP, x, static_cast<int32_t>(n));
        } else {
            ctpsv_kernel_upper_conj_unit_small<<<numBlocks, nullptr, stream>>>(AP, x, static_cast<int32_t>(n));
        }
    }
}

void ctpsv_kernel_do_small(
    uint8_t* AP, uint8_t* x, int n, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag,
    uint32_t numBlocks, void* stream)
{
    if (uplo == ACLBLAS_LOWER) {
        CtpsvKernelDoSmallLower(AP, x, n, trans, diag, numBlocks, stream);
    } else {
        CtpsvKernelDoSmallUpper(AP, x, n, trans, diag, numBlocks, stream);
    }
}

#endif // ACLBLAS_CTPSV_KERNEL_H

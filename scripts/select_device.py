#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
#
# 运行时智能选卡脚本：为测试运行挑选空闲的 NPU 设备。
#
# 用法:
#   python3 scripts/select_device.py [--prefer N] [--threshold T] [--lib PATH]
#
# 选卡逻辑:
#   1. aclrtGetDeviceCount 获取设备数；count == 0 -> 输出 NO_DEVICE, 退出码 10
#   2. 优先检查 --prefer 指定的设备（默认 0）：
#      空闲（利用率 <= 阈值）则选中，输出该设备号，退出码 0
#   3. prefer 被占用、或 prefer 越界/不存在时，按 id 升序遍历其余设备，选第一个空闲的
#   4. 全部设备被占用 -> 输出 ALL_BUSY, 退出码 20
#   5. libascendcl.so 加载失败 / ACL API 返回错误 -> 输出 ACL_ERROR, 退出码 30
#
# 输出:
#   stdout 仅输出最终结论: 成功为纯数字设备号；失败为 NO_DEVICE / ALL_BUSY / ACL_ERROR
#   stderr 输出诊断信息（各设备利用率等）
#
# 忙闲判定（利用率字段选择依据）:
#   以 aclrtGetDeviceUtilizationRate 返回的 aclrtUtilizationInfo 结构为准，其字段定义见
#   libascendcl.h（acl_rt.h）:
#       int32_t cubeUtilization;    // AI Core(cube) 利用率
#       int32_t vectorUtilization;  // AI Vector 利用率
#       int32_t aicpuUtilization;   // AI CPU 利用率
#       int32_t memoryUtilization;  // 内存利用率
#   取 cube/vector/aicpu 三个字段中"有效(>=0)"值的最大值作为判定指标（最宽口径）；
#   实测 ascend950 驱动上 memoryUtilization 返回 -1（该字段不支持，参见 ACL 文档），
#   负值字段一律按"未知"忽略，不参与判定；三个字段全部无效时按空闲(0)处理，
#   避免旧驱动/新硬件上 API 不支持导致误判全部占用。
#
#   阈值（--threshold，单位 %，0~100）默认 5：
#   - 阈值过低（如 <=3）时，驱动监控/心跳产生的偶发瞬时占用可能造成误判"忙"；
#   - 阈值过高（如 >=10）时，设备明显有负载仍被判"空闲"；
#   - 5% 为经验折中值，可按实际环境通过 --threshold 调整。
# ----------------------------------------------------------------------------------------------------------

import argparse
import ctypes
import math
import os
import sys

EXIT_OK = 0
EXIT_NO_DEVICE = 10
EXIT_ALL_BUSY = 20
EXIT_ACL_ERROR = 30

DEFAULT_LIB_FALLBACK = "/usr/local/Ascend/ascend-toolkit/latest/lib64/libascendcl.so"


class UtilizationInfo(ctypes.Structure):
    # 与 acl_rt.h 中 aclrtUtilizationInfo 定义保持一致
    _fields_ = [
        ("cubeUtilization", ctypes.c_int32),
        ("vectorUtilization", ctypes.c_int32),
        ("aicpuUtilization", ctypes.c_int32),
        ("memoryUtilization", ctypes.c_int32),
        ("utilizationExtend", ctypes.c_void_p),  # reserved, 当前版本须为 NULL
    ]


def resolve_lib_path(explicit=None):
    if explicit:
        return explicit
    home = os.environ.get("ASCEND_HOME_PATH")
    if home:
        cand = os.path.join(home, "lib64", "libascendcl.so")
        if os.path.isfile(cand):
            return cand
    return DEFAULT_LIB_FALLBACK


def load_acl(lib_path):
    lib = ctypes.CDLL(lib_path)
    lib.aclInit.argtypes = [ctypes.c_char_p]
    lib.aclInit.restype = ctypes.c_int
    lib.aclFinalize.argtypes = []
    lib.aclFinalize.restype = ctypes.c_int
    lib.aclrtGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
    lib.aclrtGetDeviceCount.restype = ctypes.c_int
    lib.aclrtGetDeviceUtilizationRate.argtypes = [ctypes.c_int32, ctypes.POINTER(UtilizationInfo)]
    lib.aclrtGetDeviceUtilizationRate.restype = ctypes.c_int
    return lib


def get_device_count(lib):
    count = ctypes.c_uint32(0)
    rc = lib.aclrtGetDeviceCount(ctypes.byref(count))
    if rc != 0:
        raise RuntimeError("aclrtGetDeviceCount failed, ret=%d" % rc)
    return count.value


def get_device_utilization(lib, device_id):
    info = UtilizationInfo(0, 0, 0, 0, None)
    rc = lib.aclrtGetDeviceUtilizationRate(device_id, ctypes.byref(info))
    if rc != 0:
        raise RuntimeError("aclrtGetDeviceUtilizationRate(%d) failed, ret=%d" % (device_id, rc))
    fields = [info.cubeUtilization, info.vectorUtilization, info.aicpuUtilization]
    valid = [v for v in fields if v >= 0]
    return float(max(valid)) if valid else 0.0


def is_busy(utilization, threshold):
    return utilization > threshold


def decide(device_count, get_util, prefer, threshold):
    """选卡决策纯函数：返回 (exit_code, device_id_or_None)，便于单元测试。

    get_util(dev) -> 该设备判定利用率（float，%）。
    """
    if device_count <= 0:
        return EXIT_NO_DEVICE, None

    order = []
    if 0 <= prefer < device_count:
        order.append(prefer)
    for dev in range(device_count):
        if dev != prefer:
            order.append(dev)

    for dev in order:
        util = get_util(dev)
        if not is_busy(util, threshold):
            return EXIT_OK, dev
    return EXIT_ALL_BUSY, None


def parse_threshold(value):
    """阈值必须是 0~100 范围内的有限百分比。"""
    threshold = float(value)
    if not math.isfinite(threshold) or not 0 <= threshold <= 100:
        raise argparse.ArgumentTypeError("threshold must be finite and between 0 and 100")
    return threshold


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Select an idle NPU device for tests (prints device id on stdout).")
    parser.add_argument("--prefer", type=int, default=0,
                        help="preferred device id (default: 0); "
                             "fall back to other idle devices if it is busy or missing")
    parser.add_argument("--threshold", type=parse_threshold, default=5.0,
                        help="busy threshold in percent, 0-100 (default: 5.0)")
    parser.add_argument("--lib", default=None,
                        help="path to libascendcl.so "
                             "(default: $ASCEND_HOME_PATH/lib64/libascendcl.so)")
    args = parser.parse_args(argv)

    lib_path = resolve_lib_path(args.lib)
    if not os.path.isfile(lib_path):
        print("ACL_ERROR")
        sys.stderr.write("[error] libascendcl.so not found: %s\n" % lib_path)
        return EXIT_ACL_ERROR

    try:
        lib = load_acl(lib_path)
    except OSError as exc:
        print("ACL_ERROR")
        sys.stderr.write("[error] failed to load %s: %s\n" % (lib_path, exc))
        return EXIT_ACL_ERROR

    initialized = (lib.aclInit(None) == 0)
    if not initialized:
        sys.stderr.write("[warn] aclInit failed, try raw aclrt calls anyway\n")

    try:
        count = get_device_count(lib)
        sys.stderr.write("[info] device count=%d, prefer=%d, threshold=%.1f%%\n"
                         % (count, args.prefer, args.threshold))

        def get_util(dev):
            util = get_device_utilization(lib, dev)
            sys.stderr.write("[info] device %d utilization=%.1f%%\n" % (dev, util))
            return util

        code, chosen = decide(count, get_util, args.prefer, args.threshold)
    except RuntimeError as exc:
        print("ACL_ERROR")
        sys.stderr.write("[error] %s\n" % exc)
        return EXIT_ACL_ERROR
    finally:
        if initialized:
            lib.aclFinalize()

    if code == EXIT_OK:
        print(chosen)
        return EXIT_OK
    if code == EXIT_NO_DEVICE:
        print("NO_DEVICE")
        return EXIT_NO_DEVICE
    print("ALL_BUSY")
    return EXIT_ALL_BUSY


if __name__ == "__main__":
    sys.exit(main())

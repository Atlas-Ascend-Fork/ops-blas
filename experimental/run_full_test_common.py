#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""TrsmBatched 系列算子全量测试的公共驱动（精度 + 性能）。

各算子脚本只需构造 Harness 并提供 CASES、golden 生成、精度校验等差异逻辑，
本模块统一负责 CANN 环境加载、shell=False 命令执行、编译、msprof 解析与结果打印。
"""
import argparse
import csv
import glob
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)


def resolve_ascend_home():
    """定位 CANN set_env.sh 所在根目录，优先环境变量，其次常见安装位置（不含个人绝对路径）。"""
    candidates = [
        os.environ.get("ASCEND_HOME_PATH"),
        os.environ.get("ASCEND_TOOLKIT_HOME"),
        "/usr/local/Ascend/ascend-toolkit/latest",
        os.path.expanduser("~/Ascend/ascend-toolkit/latest"),
    ]
    for cand in candidates:
        if cand and os.path.isfile(os.path.join(cand, "set_env.sh")):
            return cand
    return candidates[0] or ""


def _find_bash():
    """返回 bash 可执行文件的绝对路径，找不到则回退到常见位置。"""
    return shutil.which("bash") or "/bin/bash"


def load_cann_env(ascend_home):
    """一次性 source CANN set_env.sh，把结果环境变量抓成 dict 供后续 shell=False 复用。

    通过临时脚本文件执行固定逻辑（source 后 env -0 导出），不使用 shell=True，
    也不把调用方输入拼进命令，避免 shell 注入面。
    """
    base = dict(os.environ)
    base.setdefault("LD_LIBRARY_PATH", "")
    setenv = os.path.join(ascend_home, "set_env.sh")
    if not os.path.isfile(setenv):
        return base
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as fh:
        fh.write('source "$1" >/dev/null 2>&1; env -0\n')
        helper = fh.name
    try:
        p = subprocess.run(
            [_find_bash(), helper, setenv],
            capture_output=True, text=True, env=base, timeout=60)
    finally:
        os.unlink(helper)
    if p.returncode != 0:
        return base
    env = {}
    for item in p.stdout.split("\0"):
        if "=" in item:
            k, v = item.split("=", 1)
            env[k] = v
    return env or base


def write_config(project_dir, cfg_rel, lines):
    """把 config 行写入 <project_dir>/<cfg_rel>，返回相对路径。"""
    with open(os.path.join(project_dir, cfg_rel), "w") as f:
        f.write("\n".join(lines) + "\n")
    return cfg_rel


class Harness:
    """承载单个算子的全量测试流程，差异逻辑由构造参数（回调 + 常量）注入。"""

    def __init__(self, *, project_dir, test_bin, perf_out, kernel_name, cases,
                 description, fmt_mode, gen_precision, verify_case, prec_header,
                 prec_row_prefix, metric_label, gen_perf):
        self.project_dir = project_dir
        self.test_bin = test_bin
        self.perf_out = perf_out
        self.kernel_name = kernel_name
        self.cases = cases
        self.description = description
        self.fmt_mode = fmt_mode
        self.gen_precision = gen_precision
        self.verify_case = verify_case
        self.prec_header = prec_header
        self.prec_row_prefix = prec_row_prefix
        self.metric_label = metric_label
        self.gen_perf = gen_perf
        self.cann_env = load_cann_env(resolve_ascend_home())

    def run(self, argv, timeout=600, cwd=None):
        """以 shell=False 执行 argv 列表命令，环境使用已 source 好的 CANN 环境。"""
        return subprocess.run(argv, cwd=cwd or self.project_dir,
                              capture_output=True, text=True, timeout=timeout, env=self.cann_env)

    def build(self, force):
        if os.path.isfile(os.path.join(self.project_dir, self.test_bin)) and not force:
            logging.info("Build exists, skip. (use without --skip-build to rebuild)")
            return True
        logging.info("=== Building ===")
        build_dir = os.path.join(self.project_dir, "build")
        os.makedirs(build_dir, exist_ok=True)
        for argv in (["cmake", ".."], ["make", "-j8"]):
            r = self.run(argv, timeout=900, cwd=build_dir)
            if r.returncode != 0:
                logging.error((r.stderr or r.stdout)[-600:])
                return False
        logging.info("Build succeeded.")
        return True

    def parse_msprof_durations(self, prof_root):
        """解析 msprof 每次 launch 的 OpBasicInfo，返回按 launch 序号排序的 Task Duration(us) 列表。"""
        opprof = sorted(glob.glob(os.path.join(prof_root, "OPPROF_*")))
        if not opprof:
            return []
        base = os.path.join(opprof[-1], self.kernel_name)
        if not os.path.isdir(base):
            return []
        durs = []
        for name in sorted((x for x in os.listdir(base) if x.isdigit()), key=int):
            csvs = glob.glob(os.path.join(base, name, "OpBasicInfo_*.csv"))
            if not csvs:
                continue
            try:
                with open(csvs[0], newline="") as fh:
                    reader = csv.reader(fh)
                    next(reader, None)
                    row = next(reader, [])
                duration = float(row[2])
            except (OSError, IndexError, ValueError, csv.Error):
                logging.error("Cannot read Task Duration from msprof report: %s", csvs[0])
                return []
            durs.append((int(name), duration))
        durs.sort()
        return [v for _, v in durs]

    def run_precision(self):
        """一次 config 批量运行全部用例（设备只初始化一次），再逐例校验精度。"""
        logging.info("=" * 72)
        logging.info("  PRECISION")
        logging.info("=" * 72)
        cfg_rel, tags = self.gen_precision()
        if len(tags) != len(self.cases):
            logging.error("Precision: generated %d tags for %d cases", len(tags), len(self.cases))
            return False
        r = self.run([f"./{self.test_bin}", cfg_rel], timeout=900)
        if r.returncode != 0:
            logging.error((r.stdout or "")[-400:])
            logging.error((r.stderr or "")[-400:])
            return False

        logging.info(self.prec_header)
        logging.info("-" * len(self.prec_header))
        passed = failed = 0
        for i, (c, tag) in enumerate(zip(self.cases, tags)):
            mode = self.fmt_mode(c[3], c[4], c[5], c[6])
            ok, metric = self.verify_case(c, tag)
            row = self.prec_row_prefix(i, c, mode)
            if ok:
                passed += 1
                logging.info(f"{row} {'PASS':>8} {metric:>10.2e}")
            else:
                failed += 1
                logging.error(f"{row} {'**FAIL**':>8} {metric:>10.2e}")
        logging.info("-" * len(self.prec_header))
        logging.info(f"Precision: {passed} passed, {failed} failed, {len(self.cases)} total")
        return failed == 0

    def run_perf(self):
        """用一次 msprof op（config 批量模式）采集全部用例的真实 kernel Task Duration(us)。"""
        prof_dir = os.path.join(self.project_dir, self.perf_out)
        shutil.rmtree(prof_dir, ignore_errors=True)
        os.makedirs(prof_dir, mode=0o750, exist_ok=True)  # msprof 拒绝 group/other 可写目录
        os.chmod(prof_dir, 0o750)

        logging.info("=" * 72)
        logging.info("  PERFORMANCE (msprof op, real kernel Task Duration)")
        logging.info("=" * 72)
        cfg_rel, _ = self.gen_perf()
        cmd = ["msprof", "op", f"--output={self.perf_out}", f"--launch-count={len(self.cases)}",
               "--warm-up=0", f"./{self.test_bin}", cfg_rel]
        r = self.run(cmd, timeout=900)
        durs = self.parse_msprof_durations(prof_dir)
        if r.returncode != 0 or len(durs) < len(self.cases):
            logging.error((r.stderr or r.stdout)[-600:])
            logging.error(f"msprof captured {len(durs)}/{len(self.cases)} launches")
            return False

        hdr = f"{'#':>3} {'m':>5} {'n':>5} {'bat':>4} {'mode':>10} {'kernel(us)':>11}"
        logging.info(hdr)
        logging.info("-" * len(hdr))
        for i, c in enumerate(self.cases):
            mode = self.fmt_mode(c[3], c[4], c[5], c[6])
            logging.info(f"{i+1:>3} {c[0]:>5} {c[1]:>5} {c[2]:>4} {mode:>10} {durs[i]:>11.2f}")
        logging.info("-" * len(hdr))
        logging.info(f"Performance: {len(self.cases)} cases, total kernel {sum(durs):.1f}us; "
                     f"raw profiling under {self.perf_out}/")
        return True

    def main(self):
        ap = argparse.ArgumentParser(description=self.description)
        ap.add_argument("--precision", action="store_true", help="run precision only")
        ap.add_argument("--perf", action="store_true", help="run performance (msprof) only")
        ap.add_argument("--skip-build", action="store_true", help="reuse existing build/")
        args = ap.parse_args()

        do_prec = args.precision or not args.perf
        do_perf = args.perf or not args.precision

        t0 = time.time()
        if not self.build(force=not args.skip_build):
            logging.error("Build failed, aborting.")
            return 1

        prec_ok = self.run_precision() if do_prec else True
        perf_ok = self.run_perf() if do_perf else True

        logging.info("=" * 72)
        logging.info(f"DONE in {time.time()-t0:.1f}s | "
                     f"Precision: {'PASS' if prec_ok else 'FAIL' if do_prec else 'skip'} | "
                     f"Perf: {'PASS' if perf_ok else 'FAIL' if do_perf else 'skip'}")
        return 0 if (prec_ok and perf_ok) else 1

#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    https://www.cann.com/software/license/
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Shared helpers for the per-operator GTest acceptance scripts.

The gemm / csymm (and any future) acceptance scripts are near-identical:
they differ only in the operator name, the in-repo test directory layout,
the baseline CSV key columns and a couple of op-specific titles. Keeping the
shared mechanics here means a bug fix lands once instead of being copy-pasted
into every operator's verify_accuracy.py / verify_performance.py / gen_csv.py.

Each operator script keeps only:
  * its op-specific constants (OP, test_dir, titles, baseline key columns),
  * a tiny ``main()`` that delegates to one of the ``main_*`` entry points below.

Scripts reach this module via a small upward-path walk (see the bootstrap at
the top of each script) because the repo has no installed python package.
"""
import argparse
import csv
import datetime
import logging
import os
import re
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Optional

logger = logging.getLogger("bench_common")

ARCH_BY_SOC = {"ascend950": "arch35", "ascend910b3": "arch22", "ascend910b4": "arch22"}
PERF_THRESHOLD = 0.4


# --------------------------------------------------------------------------
# Operator-specific bundles (keep the shared entry points' arity <= 5, G.FNM.03)
# --------------------------------------------------------------------------
@dataclass
class Keyer:
    """Baseline-lookup key + extra report columns, operator-specific."""

    key_fn: Callable
    extra_fn: Callable
    perf_threshold: float = PERF_THRESHOLD


@dataclass
class PerfAcceptConfig:
    """Knobs for the performance-acceptance entry point (``main_perf``)."""

    op: str
    title: str
    test_dir: str
    baseline_key_fields: list
    key_fn: Callable
    extra_fn: Callable
    gpu_csv: str
    out_prefix: str
    steady_re: Optional[str] = None
    perf_threshold: float = PERF_THRESHOLD


@dataclass
class CsvSpec:
    """Column schemas + labelling for case-CSV generation (``emit_bench_csv``)."""

    fields: list
    baseline_header: list
    op_prefix: str
    baseline_fmt: Optional[Callable] = None


@dataclass
class GenSpec:
    """Generator identity for case-CSV generation (``run_gen_main``)."""

    gen_class: Callable
    plan: Callable
    op_name: str
    default_seed: int
    desc: str


# --------------------------------------------------------------------------
# Build / locate / install
# --------------------------------------------------------------------------
def find_binary(repo, test_dir, op):
    """Locate the per-operator GTest binary under build/."""
    candidates = [
        os.path.join(repo, "build", "test", test_dir, op, f"{op}_test"),
        os.path.join(repo, "build", "test", op, f"{op}_test"),
    ]
    for p in candidates:
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


def install_csv(repo, soc, csv_path, test_dir, op):
    """Copy a generated CSV into the in-repo test dir (backing up the original)."""
    arch = ARCH_BY_SOC.get(soc, "arch35")
    dst = os.path.join(repo, "test", test_dir, arch, f"{op}_test.csv")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        shutil.copy2(dst, dst + ".bak")
    shutil.copy2(csv_path, dst)
    logger.info(f"  用例安装: {csv_path} -> {dst}（原文件备份为 .bak）")


def build(repo, soc, op):
    cmd = ["bash", "build.sh", f"--soc={soc}", f"--ops={op}"]
    logger.info(f"  编译: {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=repo, timeout=600)
    if r.returncode != 0:
        logger.info("  编译失败")
        return False
    logger.info("  编译成功")
    return True


# --------------------------------------------------------------------------
# Accuracy acceptance
# --------------------------------------------------------------------------
def run_accuracy(binary, filter_str, device, timeout):
    """Run the accuracy GTest (everything but TC_PF) and parse PASS/FAIL lines."""
    env = os.environ.copy()
    if device is not None:
        env["ASCEND_DEVICE_ID"] = str(device)
    parts = [f"*{filter_str}*"] if filter_str else []
    parts.append("-*TC_PF*")   # 排除性能用例
    gf = ":".join(parts)
    cmd = [binary, f"--gtest_filter={gf}"]
    logger.info(f"  运行: {cmd[0]} --gtest_filter={gf}")
    try:
        r = subprocess.run(cmd, timeout=timeout, env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    except subprocess.TimeoutExpired:
        logger.info(f"  超时 ({timeout}s)，大尺寸用例较慢，请加大 --timeout")
        return 0, 1, [("FAIL", f"<timeout:{timeout}s>", 0)]
    details = []
    for line in r.stdout.splitlines():
        m = re.match(r"\[\s+OK\s+\]\s+\S+/(\S+)\s+\((\d+)\s*ms\)", line)
        if m:
            details.append(("PASS", m.group(1), int(m.group(2))))
            continue
        m = re.match(r"\[\s+FAILED\s+\]\s+\S+/(\S+)", line)
        if m:
            details.append(("FAIL", m.group(1), 0))
    passed = sum(1 for s, _, _ in details if s == "PASS")
    failed = sum(1 for s, _, _ in details if s == "FAIL")
    if r.returncode != 0 and failed == 0:
        details.append(("FAIL", f"<exit:{r.returncode}>", 0))
        failed += 1
    return passed, failed, details


def main_accuracy(op, title, test_dir):
    """Shared ``main()`` for accuracy acceptance (build + run + report).

    Returns the process exit code (``0`` all-pass, else ``1``); the caller's
    ``__main__`` guard performs the single ``sys.exit`` (G.ERR.11).
    """
    ap = argparse.ArgumentParser(description=title)
    ap.add_argument("--repo", required=True, help="ops-blas 仓库路径")
    ap.add_argument("--soc", required=True, help="SOC 版本，如 ascend950")
    ap.add_argument("--csv", default=None, help="先复制用例 CSV 到仓内测试目录")
    ap.add_argument("--filter", default=None, help="用例名过滤 (如 TC_L0)")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--device", type=int, default=None, help="NPU 设备 ID")
    ap.add_argument("--timeout", type=int, default=1800, help="超时秒数")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    repo = os.path.abspath(args.repo)
    logger.info(f"\n{'=' * 70}\n  {title}\n{'=' * 70}")
    if args.csv:
        install_csv(repo, args.soc, os.path.abspath(args.csv), test_dir, op)
    if not args.skip_build:
        if not build(repo, args.soc, op):
            return 1
    else:
        logger.info("  跳过编译")
    binary = find_binary(repo, test_dir, op)
    if not binary:
        logger.info("  找不到测试二进制，请先编译")
        return 1
    logger.info(f"  二进制: {binary}")
    passed, failed, details = run_accuracy(binary, args.filter, args.device, args.timeout)
    logger.info(f"\n  {'case':<50} {'result':>6} {'ms':>6}\n  {'-' * 64}")
    for status, name, ms in details:
        logger.info(f"  {name:<50} {status:>6} {ms:>6}" + ("  <==" if status == "FAIL" else ""))
    logger.info(f"\n  汇总: PASS={passed}  FAIL={failed}  (共{passed + failed})")
    if failed == 0 and passed > 0:
        logger.info("  结论: ALL PASS")
    elif failed > 0:
        logger.info(f"  结论: {failed} FAILURES")
    else:
        logger.info("  结论: NO TESTS RAN")
    return 0 if failed == 0 and passed > 0 else 1


# --------------------------------------------------------------------------
# Performance / memory acceptance
# --------------------------------------------------------------------------
def load_tc_pf_map(path):
    with open(path, newline="") as _f:
        return {r["case_name"]: r for r in csv.DictReader(_f)
                if r["case_name"].startswith("TC_PF_")}


def parse_csv_params(repo, test_dir, op):
    """Find the in-repo <op>_test.csv and return a {case_name: row} map of TC_PF rows."""
    for base in [os.path.join(repo, "test", test_dir, op), os.path.join(repo, "test", test_dir)]:
        for arch in ["arch35", "arch22"]:
            p = os.path.join(base, arch, f"{op}_test.csv")
            if os.path.exists(p):
                return load_tc_pf_map(p)
    return {}


def load_baseline(gpu_csv, key_fields):
    """Return {(key_tuple): gpu_ms}; rows with an empty gpu_ms (unfilled) are skipped.

    ``key_fields`` lists the CSV columns forming the key (e.g. ["m","n","transA","transB"]
    for gemm, ["m","n","side","uplo"] for csymm). ``m``/``n`` are coerced to int.
    """
    if not os.path.exists(gpu_csv):
        return {}
    out = {}
    with open(gpu_csv, newline="") as _f:
        for r in csv.DictReader(_f):
            if r["gpu_ms"].strip():
                key = tuple(int(r[c]) if c in ("m", "n") else r[c] for c in key_fields)
                out[key] = float(r["gpu_ms"])
    return out


def parse_results(stdout_text, steady_re=None):
    """Parse GTest output into per-case results.

    ``steady_re`` (if given) overrides a case's wall-clock with a steady-state
    kernel time (csymm emits ``[CSYMM_PERF]``).
    """
    results = []
    steady = {}
    for line in stdout_text.splitlines():
        if steady_re is not None:
            m = re.search(steady_re, line)
            if m:
                steady[m.group(1)] = float(m.group(2)) / 1000.0  # us -> ms
                continue
        m = re.match(r"\[\s+OK\s+\]\s+\S+/(\S+)\s+\((\d+)\s*ms\)", line)
        if m:
            results.append({"case_name": m.group(1), "npu_ms": float(m.group(2)), "status": "ok"})
            continue
        m = re.match(r"\[\s+FAILED\s+\]\s+\S+/(\S+)", line)
        if m:
            results.append({"case_name": m.group(1), "npu_ms": -1.0, "status": "failed"})
    for res in results:
        if res["case_name"] in steady:
            res["npu_ms"] = steady[res["case_name"]]
    return results


def summarize(results, csv_params, baseline, keyer):
    """Compute the speedup ratio and verdict for each result row.

    ``keyer`` bundles the operator-specific callbacks: ``key_fn(params, m, n)``
    builds the baseline lookup key and ``extra_fn(params)`` returns the extra
    report columns (e.g. transA/transB or side/uplo).
    """
    out_rows = []
    for res in results:
        p = csv_params.get(res["case_name"], {})
        m_val = int(p.get("m", "0"))
        n_val = int(p.get("n", "0"))
        base = baseline.get(keyer.key_fn(p, m_val, n_val))
        npu = res["npu_ms"]
        ratio = 0
        if res.get("status") == "failed":
            verdict = "FAIL"
        elif base is not None and npu > 0:
            ratio = base / npu
            verdict = "PASS" if ratio >= keyer.perf_threshold else "FAIL"
        else:
            verdict = "NO_REF"
        row = {
            "case_name": res["case_name"], "m": m_val, "n": n_val,
            "npu_ms": (f"{npu:.3f}" if npu > 0 else ""),
            "baseline_ms": (f"{base:.3f}" if base else ""),
            "ratio": round(ratio, 3) if ratio else "",
            "verdict": verdict,
        }
        row.update(keyer.extra_fn(p))
        out_rows.append(row)
    return out_rows


def print_report(out_csv, out_rows):
    """Write the result CSV and print the summary; return the FAIL count."""
    if out_rows:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            for row in out_rows:
                w.writerow(row)
        logger.info(f"  结果: {out_csv} ({len(out_rows)} 条)")
    n_pass = sum(1 for r in out_rows if r["verdict"] == "PASS")
    n_fail = sum(1 for r in out_rows if r["verdict"] == "FAIL")
    n_noref = sum(1 for r in out_rows if r["verdict"] == "NO_REF")
    logger.info(f"\n  {'case':<16} {'size':>12} {'NPU_ms':>10} {'base_ms':>10} {'ratio':>6} {'verdict':>6}")
    logger.info(f"  {'-' * 60}")
    for r in out_rows:
        logger.info(f"  {r['case_name']:<16} {str(r['m']) + 'x' + str(r['n']):>12} "
                    f"{str(r['npu_ms']):>10} {str(r['baseline_ms']):>10} "
                    f"{str(r['ratio']):>6} {r['verdict']:>6}")
    logger.info(f"\n  汇总: PASS={n_pass}  FAIL={n_fail}  NO_REF={n_noref}")
    logger.info(f"  注: GTest 输出耗时含 host 准备+kernel+golden 计算+比对，为保守上界；")
    logger.info(f"      精确 kernel 耗时可配合 msprof 采集。")
    return n_fail


def main_perf(cfg):
    """Shared ``main()`` for performance acceptance (build + run TC_PF + report).

    ``cfg`` is a :class:`PerfAcceptConfig`; returns the process exit code so the
    caller's ``__main__`` guard owns the single ``sys.exit`` (G.ERR.11).
    """
    ap = argparse.ArgumentParser(description=cfg.title)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--soc", required=True)
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--timeout", type=int, default=3600)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    repo = os.path.abspath(args.repo)
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(os.path.dirname(cfg.gpu_csv), f"{cfg.out_prefix}_{ts}.csv")
    logger.info(f"\n{'=' * 90}")
    logger.info(f"  {cfg.title}")
    logger.info(f"  验收标准: NPU 耗时 <= GPU H100 耗时 / {cfg.perf_threshold} "
                f"(倍率 >= {cfg.perf_threshold})")
    logger.info(f"{'=' * 90}")
    if not args.skip_build:
        if not build(repo, args.soc, cfg.op):
            return 1
    else:
        logger.info("  跳过编译")
    binary = find_binary(repo, cfg.test_dir, cfg.op)
    if not binary:
        logger.info("  找不到测试二进制")
        return 1
    logger.info(f"  二进制: {binary}")
    csv_params = parse_csv_params(repo, cfg.test_dir, cfg.op)
    baseline = load_baseline(cfg.gpu_csv, cfg.baseline_key_fields)
    logger.info(f"  基线: {cfg.gpu_csv}（已回填 {len(baseline)} 条）")
    if not baseline:
        logger.info("  [提示] 基线尚未回填，全部用例将标记 NO_REF，仅采集 NPU 耗时")
    env = os.environ.copy()
    if args.device is not None:
        env["ASCEND_DEVICE_ID"] = str(args.device)
    cmd = [binary, "--gtest_filter=*TC_PF*"]
    logger.info(f"  运行: {cmd[0]} --gtest_filter=*TC_PF*")
    try:
        r = subprocess.run(cmd, timeout=args.timeout, env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    except subprocess.TimeoutExpired:
        logger.info(f"  超时 ({args.timeout}s)，建议加大 --timeout")
        return 1
    results = parse_results(r.stdout, steady_re=cfg.steady_re)
    keyer = Keyer(cfg.key_fn, cfg.extra_fn, cfg.perf_threshold)
    out_rows = summarize(results, csv_params, baseline, keyer)
    n_fail = print_report(out_csv, out_rows)
    if r.returncode != 0:
        logger.info(f"  测试进程异常退出 (returncode={r.returncode})")
    return 0 if r.returncode == 0 and n_fail == 0 and len(out_rows) > 0 else 1


# --------------------------------------------------------------------------
# CSV generation
# --------------------------------------------------------------------------
def _baseline_row(i, r, spec):
    """Build one ``gpu_baseline.csv`` placeholder row (``gpu_ms`` left blank)."""
    prefix = f"{spec.op_prefix}-base" if i < 4 else f"{spec.op_prefix}-perf"
    row = [f"{prefix}-{i:03d}"]
    fmt = spec.baseline_fmt
    for c in spec.baseline_header[1:-1]:
        val = r.get(c, "")
        row.append(fmt(c, val) if fmt is not None else val)
    row.append("")  # gpu_ms，待基线测试后回填
    return row


def _write_baseline_csv(bl_out, pf_rows, spec):
    """Write ``gpu_baseline.csv`` placeholder rows (``gpu_ms`` left blank for backfill)."""
    with open(bl_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(spec.baseline_header)
        for i, r in enumerate(pf_rows):
            w.writerow(_baseline_row(i, r, spec))


def emit_bench_csv(g, out, bl_out, spec):
    """Write ``<op>_test.csv`` and ``gpu_baseline.csv``; reject non-compact perf rows.

    ``spec`` is a :class:`CsvSpec`: ``fields`` is the full column list for the case
    CSV, ``baseline_header`` the full column list for ``gpu_baseline.csv``
    (``id``/``gpu_ms`` are filled here, every other column comes from the perf-row
    dict), ``op_prefix`` labels the baseline ids (e.g. ``gemm``/``csymm``) and the
    optional ``baseline_fmt(col, value)`` rewrites a baseline data column (csymm
    strips the ``ACLBLAS_`` prefix from side/uplo; gemm leaves transA/transB alone).
    """
    acc_rows = [r for r in g.rows if not r["case_name"].startswith("TC_PF")]
    pf_rows = [r for r in g.rows if r["case_name"].startswith("TC_PF")]
    # 性能用例纯连续校验：非紧凑前导维会污染性能与内存基线（功能维度由精度用例覆盖）
    for r in pf_rows:
        if r["lda"] or r["ldb"] or r["ldc"]:
            raise RuntimeError(f"性能用例含非紧凑前导维: {r['case_name']}")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=spec.fields)
        w.writeheader()
        for r in g.rows:
            w.writerow(r)
    _write_baseline_csv(bl_out, pf_rows, spec)
    cats = Counter(r["case_name"].split("_")[1] for r in acc_rows)
    logger.info(f"{out}: 共 {len(g.rows)} 条（精度 {len(acc_rows)} + 性能/内存 {len(pf_rows)}）")
    logger.info("精度用例分类: %s", dict(sorted(cats.items())))
    logger.info(f"{bl_out}: {len(pf_rows)} 条基线占位（gpu_ms 待回填）")


def run_gen_main(gen_spec, spec):
    """Shared ``main()`` for case-CSV generation.

    ``gen_spec`` (a :class:`GenSpec`) carries the generator class, its
    ``plan(g, accuracy, perf)`` populator, the op name / seed / description;
    ``spec`` (a :class:`CsvSpec`) drives ``emit_bench_csv``. Returns the process
    exit code (caller's ``__main__`` guard owns ``sys.exit``).
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=gen_spec.desc)
    ap.add_argument("--accuracy", type=int, default=1000, help="精度用例条数（默认1000）")
    ap.add_argument("--perf", type=int, default=200, help="性能/内存用例条数（默认200）")
    ap.add_argument("--seed", type=int, default=gen_spec.default_seed,
                    help="随机种子（固定可复现）")
    ap.add_argument("--dist", choices=["uniform", "mixed"], default="uniform",
                    help="输入分布；mixed 依赖测试工程扩展，暂以均匀生成")
    ap.add_argument("--out", default=None)
    ap.add_argument("--baseline-out", default=None)
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    out = args.out or os.path.join(here, f"{gen_spec.op_name}_test.csv")
    bl_out = args.baseline_out or os.path.join(here, "gpu_baseline.csv")
    if args.dist == "mixed":
        logger.info("[提示] 正态分布依赖 ops-blas 测试工程扩展（另行任务），本次以均匀分布生成")
    g = gen_spec.gen_class(args.seed)
    gen_spec.plan(g, args.accuracy, args.perf)
    try:
        emit_bench_csv(g, out, bl_out, spec)
    except RuntimeError as e:
        logger.error(str(e))
        return 1
    return 0

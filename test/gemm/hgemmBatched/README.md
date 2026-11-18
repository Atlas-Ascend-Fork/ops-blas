# aclblasHgemmBatched test

The tests in this directory validate the `aclblasHgemmBatched` interface on Ascend 950PR (`arch35`).
`aclblasHgemmBatched` computes, per batch `i`:

```
C[i] = alpha * op(A[i]) * op(B[i]) + beta * C[i],  i ∈ [0, batchCount)
```

where `A[i]`, `B[i]`, `C[i]` are FLOAT16 column-major matrices, `alpha`/`beta` are FLOAT16 scalar
pointers, and `op()` is one of `N`/`T`/`C` (for real FLOAT16, `T` and `C` are equivalent).

## Directory structure

```text
test/gemm/hgemmBatched/
├── CMakeLists.txt
├── README.md
├── hgemmBatched_golden.h
├── hgemmBatched_param.h
├── run_self_test.py
├── run_memcheck.py
└── arch35/
    ├── hgemmBatched_npu_wrapper.h
    ├── hgemmBatched_test.cpp
    └── hgemmBatched_test.csv
```

- `hgemmBatched_param.h` — CSV row parser (`transA/transB`, `m/n/k`, `alpha`, `aarray_fill/lda`,
  `barray_fill/ldb`, `beta`, `carray_fill/ldc`, `batchCount`, `expect_result`, `random_seed`).
- `hgemmBatched_golden.h` — FLOAT32 CPU reference (`cblas_sgemm`) per batch; the result is rounded to
  FLOAT16 (IEEE 754 round-to-nearest-even) before comparison.
- `arch35/hgemmBatched_npu_wrapper.h` — host-to-device buffer management and the `*_npu` call adapter.
- `arch35/hgemmBatched_test.cpp` — GTest cases: a null-handle test, CSV-driven accuracy cases, and
  performance cases (`PerformanceContract` + optional CSV `TC_PF_*`).
- `run_self_test.py` — CSV completeness + execution gate (self-test evidence).
- `run_memcheck.py` — `mssanitizer` memcheck memory-safety gate (memory-safety evidence).

The CSV covers all `N/T/C` transpose combinations, `m/n/k` size scans (0, 1, primes, powers of two ±1,
unaligned, up to 2048 for the shipped set; generator supports up to 16384), `batchCount` scans
(0, 1, 2, primes, powers of two, unaligned, up to 1024 in the shipped set; generator up to 65536),
special scalar values (0, ±1, large values near 65504), leading-dimension padding, boundary/negative
cases (`m=0`/`n=0`/`batchCount=0` no-op, `k=0` beta scaling, null scalar/array pointers, invalid
enum/dimension/leading-dimension, negative dims, null handle), and Inf/NaN.

## Precision

Golden is FLOAT32 `cblas_sgemm` per batch, rounded to FLOAT16, then compared element-wise with
`MIXED_TOLERANCE` for FLOAT16 (`rtol = atol = 2^-10`, `required_matched_ratio = 0.99`,
`max_abs_error <= max(1e-1, 32·ULP)`). `k == 0` / `alpha == 0` degenerate cases use the beta-scaling
reference (beta == 0 zero-fills). NaN/Inf follow the task rule (NaN↔NaN, same-sign Inf↔Inf, and
NaN↔±Inf are treated as equivalent).

## Build and run

From the repository root, build the operator and its test target:

```bash
bash build.sh --soc=ascend950 --ops=hgemmBatched
```

Run all accuracy cases (the performance cases are skipped unless enabled):

```bash
./build/test/gemm/hgemmBatched/hgemmBatched_test
```

Run only the §3.3 typical performance contract:

```bash
./build/test/gemm/hgemmBatched/hgemmBatched_test --gtest_filter='*PerformanceContract*'
```

Run the full CSV performance sweep (`TC_PF_*`, warmup 5 + 60 samples, mean):

```bash
HGEMM_BATCHED_RUN_ALL_PERF=1 \
  ./build/test/gemm/hgemmBatched/hgemmBatched_test --gtest_filter='*TC_PF*'
```

For kernel-level `task duration` (msprof op mode, 5 warmup + ≥10 valid samples), profile a single
`TC_PF` case in isolation, e.g.:

```bash
msprof --output=./prof --task-time=on \
  --application='bash -c "cd build/test/gemm/hgemmBatched && HGEMM_BATCHED_RUN_ALL_PERF=1 ./hgemmBatched_test --gtest_filter=*/TC_PF_1001"'
```

## Completeness and memory-safety gates

Both scripts require a fresh output directory and return non-zero on any failure, skip, timeout or
executed-name/count mismatch. They are the self-test and memcheck evidence for acceptance (in line with
the other batched GEMM test projects).

Self-test gate — checks that the CSV holds exactly 1227 unique rows (1002 accuracy-success + 200
`TC_PF_*` performance shapes + 25 negative/boundary), then runs every `CsvDriven` accuracy case plus the
`CsvCaseCount`/`NullHandle` contract tests, requiring zero failures and zero skips:

```bash
python3 test/gemm/hgemmBatched/run_self_test.py \
  --binary build/test/gemm/hgemmBatched/hgemmBatched_test \
  --output /tmp/hgemmBatched-self-test
```

Memory-safety gate — build a separate sanitizer tree first (matching the other batched projects), then
run the memcheck queue (all negative/boundary contracts, NN/TN/NT/TT representative shapes including the
FLOAT16 fixpipe direct copy-out path, the tiny and 1024-batch boundaries, and the beta==0 zero-write
cases with a NaN-filled C). Every kernel must show a paired `Start memcheck sanitizer` /
`Sanitizer finished ... No error detected.` record with no diagnostics:

```bash
cmake -S . -B build-sanitize \
  -DSOC_VERSION=ascend950 -DASCEND_CANN_PACKAGE_PATH=$ASCEND_HOME \
  -DBUILD_TEST=ON -DTEST_NAMES=hgemmBatched \
  -DCMAKE_ASC_FLAGS="-g --cce-enable-sanitizer"
cmake --build build-sanitize -j
python3 test/gemm/hgemmBatched/run_memcheck.py \
  --binary build-sanitize/test/gemm/hgemmBatched/hgemmBatched_test \
  --output /tmp/hgemmBatched-memcheck
```

Results are written to `self_test_summary.json` and `memcheck_summary.json`.

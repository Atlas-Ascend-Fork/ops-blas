# Cdotu Test - Ascend C Kernel

## Overview
This test implements the `aclblasCdotu` operator on Ascend 950PR (arch35) using Ascend C kernel.
The operator computes the unconjugated complex dot product of two complex64 vectors:
`result = Σ(x[k] × y[j])` (i = 1..n, k = 1+(i-1)*incx, j = 1+(i-1)*incy), output a single
complex scalar. Semantics align with cuBLAS `cublasCdotu` / Netlib `cdotu` (no conjugation).

## Test Structure
- `test/dot/cdotu/cdotu_param.h` - Parameter structure (n / x / incx / y / incy / result_fill)
- `test/dot/cdotu/cdotu_golden.h` - CPU reference (double-precision near-exact + float32 special-value variant)
- `test/dot/cdotu/cdotu_npu_wrapper.h` - NPU wrapper using `aclblasCdotu` API
- `test/dot/cdotu/arch35/cdotu_test.cpp` - GTest parameterized test (1201 CSV cases)
- `test/dot/cdotu/arch35/cdotu_test.csv` - Test case definitions
- `test/dot/cdotu/CMakeLists.txt` - CMake configuration for test build

## Test Cases
The CSV (`cdotu_test.csv`) contains 1201 cases with columns:
- `case_name` / `description`: Test case identifier / description
- `n`: Vector length (0,1,primes, powers of 2 ±1, up to 2^22)
- `x` / `y`: Fill modes (RANDOM_NORM_5_5, RANDOM_ALTER, RANDOM_EXTREME, VALUE_NORM_0/INF/NAN, NULLPTR)
- `incx` / `incy`: Strides (nonzero, can be negative; 0 → INVALID_VALUE)
- `result_fill`: Result buffer mode (VALUE_NORM_0 valid, NULLPTR negative)
- `expect_result`: Expected aclblasStatus_t
- `mere_threshold` / `mare_multiplier` / `random_seed`: Precision / reproducibility

Coverage categories:
- L0 basic (8), SQ size scan (38), INC stride combos (36), FL fill (12),
  ED edge/negative (10), EX extended (897), PF perf (200)

## Build Instructions
Run the commands from the repository root. Set `CANN_HOME` to the directory containing `set_env.sh`.

```bash
CANN_HOME=/usr/local/Ascend/cann-9.1.0
# Source CANN environment
source "$CANN_HOME/set_env.sh"

cmake -S . -B build \
  -DSOC_VERSION=ascend950 \
  -DASCEND_CANN_PACKAGE_PATH="$CANN_HOME" \
  -DBUILD_TEST=ON \
  -DTEST_NAMES=cdotu \
  -DTEST_DEVICE_ID=0

cmake --build build --target cdotu_test -j
```

## Test Execution
```bash
./build/test/dot/cdotu/cdotu_test
```

## Expected Result
`[  PASSED  ] 1202 tests.` (verified on Ascend 950PR, CANN 9.1.0)

## Precision Verification
- Output is a COMPLEX64 scalar: real/imag verified separately per FLOAT32 standard
  (task §3.2): `|actual-golden| ≤ atol + rtol×|golden|` with rtol=2^-10, atol=2^-16,
  max_abs_error ≤ 1e-2 或 32×ULP.
- Golden accumulates in double precision (near-exact, independent of summation order) so a
  correct float32 tree-reduce kernel sits within the 32×ULP bound even at n~2^22.
- Extreme-value cases (FLT_MAX/FLT_MIN overflow): use float32 sequential reference (cblas
  semantics) so both golden and kernel overflow to Inf/NaN consistently; Inf/NaN judged by
  consistency (both NaN, or same-sign Inf), per task §3.2 note.
- quick return: n=0 no-op writes result=(0,0) and returns success; result==nullptr always
  returns ACLBLAS_STATUS_INVALID_VALUE (aligned with arch22 cdot), even when n==0; n<0 /
  incx=0 / incy=0 / null x,y (n>0) return ACLBLAS_STATUS_INVALID_VALUE.

## Test Steps (reproducible)
1. From the repository root, set `CANN_HOME` to your CANN installation directory and load the environment: `CANN_HOME=/usr/local/Ascend/cann-9.1.0 && source "$CANN_HOME/set_env.sh"`
2. Configure: `cmake -S . -B build -DSOC_VERSION=ascend950 -DASCEND_CANN_PACKAGE_PATH="$CANN_HOME" -DBUILD_TEST=ON -DTEST_NAMES=cdotu -DTEST_DEVICE_ID=0`
3. Build: `cmake --build build --target cdotu_test -j`
4. Run: `./build/test/dot/cdotu/cdotu_test`
5. Expect: `[  PASSED  ] 1202 tests.`
# aclblasCgemm test

The tests in this directory validate the `aclblasCgemm` interface on Ascend 950PR/950DT (`arch35`) and on Atlas A2/A3 (`arch22`).

The two architectures are compiled in isolation (see `ops_blas_get_test_target_sources` in `cmake/test.cmake`): `--soc=ascend950` compiles only `arch35/`, `--soc=ascend910b3` / `ascend910b4` compiles only `arch22/`, so each side ships its own wrapper and CSV.

## Directory structure

```text
test/gemm/cgemm/
├── CMakeLists.txt
├── README.md
├── cgemm_golden.h
├── cgemm_param.h
├── arch22/
│   ├── cgemm_npu_wrapper.h
│   ├── cgemm_test.cpp
│   └── cgemm_test.csv
└── arch35/
    ├── cgemm_npu_wrapper.h
    ├── cgemm_test.cpp
    ├── cgemm_test.csv
    └── cgemm_test_full.csv
```

The full CSV covers column-major complex matrix multiplication, all `N/T/C` transpose combinations,
non-contiguous leading dimensions, complex `alpha` and `beta`, boundary values, and invalid parameters. The CPU
reference result is computed with CBLAS and compared with the NPU result by the common test verifier.

`arch35/cgemm_test.csv` contains the first 50 cases from `arch35/cgemm_test_full.csv`,
in their original order. The test executable runs these 50 CSV cases and the separate null-handle test.
The full CSV retains all 1000 cases for reference and is not loaded by the default test entry.

## arch22 coverage

`arch22/cgemm_test.csv` holds 74 cases for Atlas A2/A3: the full `transa` x `transb` matrix over `N/T/C`, square
sizes 1..256, rectangular shapes including degenerate `m/n/k == 1`, complex `alpha`/`beta`, the `alpha=0` and `k=0`
degenerate paths, `alpha=0 & beta=1` bit-exact no-op, padded leading dimensions, and invalid-parameter return codes.

Besides the CSV, `arch22/cgemm_test.cpp` adds independent cases for: null handle, invalid enum / negative dimension /
leading dimension / null pointer, `m=0` and `n=0` no-ops, host-side `alpha`/`beta` (the wrapper uploads them, so the
device-scalar read-back branch is exercised), non-finite incoming `C` surviving a zero `beta` component (`0 * Inf` must
not become NaN), and a zero `alpha` imaginary component skipping the multiply.

## Build and run

From the repository root, build the operator and its test target:

```bash
bash build.sh --soc=ascend950 --ops=cgemm
```

Run the accuracy cases from the generated test executable in the build output directory:

```bash
./build/test/gemm/cgemm/cgemm_test
```

For Atlas A2/A3:

```bash
bash build.sh --soc=ascend910b3 --ops=cgemm
./build/test/gemm/cgemm/cgemm_test
```

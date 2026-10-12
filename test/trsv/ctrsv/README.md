# ctrsv tests

The `arch22` test is a CSV-driven GTest for `aclblasCtrsv` on Atlas 800I
A2/A3 (910B3/910B4). The supplied `arch22/ctrsv_test.csv` contains the
accuracy and `TC_PF` performance cases from the task package.

Additional `arch22` sources are collected by the shared `ctrsv_test*.cpp`
pattern: `ctrsv_test_scalar.cpp` covers scalar arithmetic and
`ctrsv_test_boundaries.cpp` covers diagonal range and dispatch boundaries,
including padded leading dimensions, negative strides, and untouched vector
gaps. Their direct GTests are included in the accuracy command below.

Build the operator and test with:

```bash
bash build.sh --soc=ascend910b3 --ops=ctrsv
```

Run accuracy cases (excluding performance cases) with:

```bash
ASCEND_DEVICE_ID=0 build/test/trsv/ctrsv/ctrsv_test --gtest_filter='*-*TC_PF*'
```

Run performance cases with the normal warmup/sample configuration used by
the task test scripts:

```bash
ASCEND_DEVICE_ID=0 build/test/trsv/ctrsv/ctrsv_test --gtest_filter='*TC_PF*'
```

## Netlib Correctness Reference

The complex reference calls Netlib BLAS `ctrsv_` directly. It does not use a
handwritten triangular solve or the OpenBLAS implementation. Configure
`CTRSV_NETLIB_LIBRARY` to an absolute path to the Netlib shared library. The
adapter uses the GNU Fortran LP64 ABI (32-bit integers and trailing `size_t`
character lengths), and rejects libraries exporting `openblas_get_config`.
The first numerical correctness case prints the resolved library path.

For the local Release/O2 build on 910B:

```bash
source /usr/local/Ascend/cann-9.1.0-beta.3/set_env.sh
export LINUX_INCLUDE_PATH=/usr/local/Ascend/cann-9.1.0-beta.3/aarch64-linux/include
export EAGER_LIBRARY_PATH=/usr/local/Ascend/cann-9.1.0-beta.3/aarch64-linux/lib64
cmake -S . -B build-o2 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_ASC_FLAGS_RELEASE='-O2 -DNDEBUG' \
  -DCTRSV_NETLIB_LIBRARY=/lib64/libblas.so.3.12.0
cmake --build build-o2 --target ctrsv_test -j8
CTRSV_TEST_CSV=/absolute/path/to/official/ctrsv_test.csv \
  ASCEND_DEVICE_ID=0 build-o2/test/trsv/ctrsv/ctrsv_test \
  --gtest_filter='Trsv/CtrsvArch22Test.CsvDriven/*-*TC_PF_*' \
  --gtest_output=xml:accuracy_netlib.xml
```

`CTRSV_TEST_CSV` overrides the CSV input without modifying the repository's
local cases. The filter selects only CSV correctness cases, excluding both
performance cases and additional direct GTests. Invalid-argument cases check
the ACLBLAS return code against the CSV; valid numerical cases call Netlib
and retain the existing MERE/MARE thresholds and nonfinite-value checks.

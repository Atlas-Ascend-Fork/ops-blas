# aclblasCgemm3mStridedBatched tests

The Ascend 950PR (`arch35`) tests use the repository GTest/CSV framework.
The supplied CSV contains accuracy and boundary cases (`TC_L0`, `TC_ED` and other
non-`TC_PF` groups) and performance cases (`TC_PF`).

## Build

Use CANN 9.1.0 and Netlib BLAS 3.10.0 for the CBLAS reference. Configure
`REFBLAS_INCLUDE_DIR`, `REFBLAS_LIB` and `REFLAPACK_LIB` when these libraries are
outside the standard search paths.

From the repository root:

```bash
source <cann-install>/set_env.sh
bash build.sh --soc=ascend950 --ops=cgemm3m_strided_batched
```

## Accuracy and boundaries

Expose one device per process and use logical device 0:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
export BLAS_DEVICE_ID=0
./build/test/gemm/cgemm3m_strided_batched/cgemm3m_strided_batched_test \
  --gtest_filter='*.Accuracy/*' --gtest_output=xml:accuracy.xml
```

The reference uses Netlib `cblas_cgemm` for each batch. Real and imaginary
components are checked separately over the complete output matrix, using
`atol=rtol=2^-13`, a matched ratio of at least 0.99, and the task's
`max(1e-2, 32*ULP)` bound. Nonfinite values follow the task's separate comparison
rules. The tests also check padding, guards and bitwise repeatability.

The alpha-zero/null-A boundary follows attachment case `TC_ED_313`: it returns
SUCCESS and performs the required C scaling without reading A.

Invalid transa/transb return INVALID_ENUM, including for empty output. This
updates the supplied TC_ED_319/320 expectations to the reviewed API contract.
When m, n or batchCount is zero, A/B/C may be null; handle, operation, shape,
leading dimensions and alpha/beta are still validated before quick return.
The TC_RG_1204-1214 cases cover these null-pointer and validation-order boundaries.

## Performance

The performance entry executes ten API calls, synchronizing after each one.
Select one CSV case per capture:

```bash
msprof op --warm-up=5 --kernel-name='*' --launch-count=5000 \
  --aic-metrics=BasicInfo --output=profile \
  ./build/test/gemm/cgemm3m_strided_batched/cgemm3m_strided_batched_test \
  --gtest_filter='*.Performance/TC_PF_1001'
```

Sum all kernel task durations belonging to each complete API call, including
every stage and batch, then average the ten valid samples. Check the actual
record count against the selected path; `--launch-count` counts kernels,
not API calls. Incomplete or failed captures are excluded. Large batches may
require collection by complete API ranges to stay within the profiler's
export capacity.

Compare with the task's GPU baseline: every case requires a speedup of at
least 0.4, and the arithmetic mean across performance cases requires at least
0.5. Record the collection mode, device, software versions and tested revision
alongside the results.

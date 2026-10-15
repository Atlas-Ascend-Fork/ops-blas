# Ctpmv arch22 validation

The implementation targets Atlas 800I A2/A3 (910B3, DAV_2201/arch22) and is
organized as one common device implementation and one batched implementation.
Host dispatch selects five path families:

| Path family | Scope |
| --- | --- |
| `strict_scalar` | Strided, short OP_N, and order-sensitive layouts; shared exceptional-value fallback |
| `upper_transpose_panel` | Short upper transpose and a single-element upper OP_N |
| `batched_reference` | Remaining transpose/conjugate layouts |
| `column_panel_notrans` | Medium OP_N; panel producers and disjoint row reducers |
| `column_partials` | Large OP_N; upper/lower template instantiations share publication and reduction |

The column-panel path also covers the former row-tile and lower-resident
fallbacks. Panel ownership is derived from the actual core count; each
reducer visits every assigned row tile, including launches with one core.
Both triangles also share column loading and accumulation in the partials path.
The transpose slab holds all 4096 supported elements, replacing the long-column
fallback with the common batched traversal. UB usage is checked at compile time.
Kernel entry templates specialize the path without modifying shared tiling
arguments. The ordered upper/lower traversal shares one template per transpose
mode, preserving the reference accumulation sequence. Short contiguous OP_N uses
the same ordered entry with a direct two-element update. The short transpose
panel shares one product template between the multiply and cross-term stages.

## Build

```bash
cmake -S . -B build -DRUN_MODE=npu -DSOC_VERSION=ascend910b3 \
  -DBUILD_TEST=ON -DTEST_NAMES=ctpmv -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS_RELEASE='-O2 -DNDEBUG' \
  -DCMAKE_ASC_FLAGS_RELEASE='-O2 -DNDEBUG'
cmake --build build --target ctpmv_test -j$(nproc)
```

## Correctness

Run the complete non-performance set on device 0:

```bash
env -u CTPMV_PERF_WARMUP -u CTPMV_PERF_ITERATIONS \
  build/test/tpmv/ctpmv/ctpmv_test --gtest_filter='-*TC_PF_*' \
  --gtest_color=no --device=0
```

The CSV contains the 1000 functional and special-value cases supplied with
the task, 200 performance cases, 72 transpose boundary regressions, and 36 non-transpose path-merge boundaries.
Run without benchmark environment variables to check all 1309 tests,
including the standalone null-handle test. The added cases cover both
triangles and T/C around the short-panel, former 2048-element slab boundary, and maximum 4096-element size. The task acceptance tolerance is absolute error `1e-2` or 32 ULP;
the test harness additionally reports stricter MERE/MARE diagnostics.

## Performance

Use more than 50 measured iterations after warmup:

```bash
CTPMV_PERF_WARMUP=10 CTPMV_PERF_ITERATIONS=100 \
  build/test/tpmv/ctpmv/ctpmv_test --gtest_filter='*TC_PF_*' \
  --gtest_color=no --device=0
```

The measured interval contains all host API calls followed by one stream
synchronization. Dividing the interval by the iteration count reports the
average end-to-end throughput, including host launch/scheduling overhead and
every device task produced by the batch. Buffer allocation and the single H2D
preparation are outside the timed region. The output lines are of the form
`[PERF] case=... average_us=...`; acceptance uses
`NPU_us <= GPU_ms * 1000 / 0.8`. The test executable prints each
`[PERF]` record directly, and the pipeline collects these records from the
performance test.



Every invocation receives an independent copy of the original input vector.
This prevents repeated in-place multiplication from changing the input range
or overflowing during warmup. The last sampled output is checked against the
same CBLAS reference as the correctness tests, after the timed interval.
Kernel dispatch and exceptional-value checks do not depend on benchmark mode.

For the P11 protocol, collect `msprof` Task Duration, discard the first 10
kernel tasks per case, and average the next 100. Compare this device time with
`GPU_ms * 1000 / 0.8`; keep it separate from the host batch average above.

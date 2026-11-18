# Chemv 950PR memcheck evidence

Device: Ascend950PR. Toolkit: CANN 9.1.0. Run date: 2026-11-07. Source: `3b6e573c4241214c04f0e746d6b2f9e8666333fd`.

The independent test build used `-g --cce-enable-sanitizer` for the Chemv kernel. The run used `mssanitizer --tool=memcheck --leak-check=yes`.

The archived summary reports 15 passing groups and `checked_kernels = 14`. The table projects group status from that summary. The excerpts below are selected verbatim from the corresponding device logs.

| Group | Status | Checked kernels |
|---|---|---:|
| contracts | PASS | 0 |
| upper_reg_tail | PASS | 1 |
| lower_reg_tail | PASS | 1 |
| upper_reg_middle | PASS | 1 |
| lower_reg_middle | PASS | 1 |
| upper_reg_large | PASS | 1 |
| lower_reg_large | PASS | 1 |
| positive_stride_padding | PASS | 1 |
| reg_padding | PASS | 1 |
| negative_stride | PASS | 1 |
| alpha_zero_beta_zero | PASS | 1 |
| alpha_zero_beta_other | PASS | 1 |
| beta_one_compute | PASS | 1 |
| upper_ex_tail | PASS | 1 |
| lower_ex_tail | PASS | 1 |

## contracts

```text
[  PASSED  ] 124 tests.
```

## upper_reg_tail

```text
[mssanitizer] Start memcheck sanitizer on kernel "chemv_kernel(unsigned char*, unsigned char*, unsigned char*, ChemvTilingData)"
[mssanitizer] Sanitizer finished on kernel "chemv_kernel(unsigned char*, unsigned char*, unsigned char*, ChemvTilingData)". No error detected.
[  PASSED  ] 1 test.
```

## lower_reg_tail

```text
[mssanitizer] Start memcheck sanitizer on kernel "chemv_kernel(unsigned char*, unsigned char*, unsigned char*, ChemvTilingData)"
[mssanitizer] Sanitizer finished on kernel "chemv_kernel(unsigned char*, unsigned char*, unsigned char*, ChemvTilingData)". No error detected.
[  PASSED  ] 1 test.
```

## upper_reg_middle

```text
[mssanitizer] Start memcheck sanitizer on kernel "chemv_kernel(unsigned char*, unsigned char*, unsigned char*, ChemvTilingData)"
[mssanitizer] Sanitizer finished on kernel "chemv_kernel(unsigned char*, unsigned char*, unsigned char*, ChemvTilingData)". No error detected.
[  PASSED  ] 1 test.
```

## lower_reg_middle

```text
[mssanitizer] Start memcheck sanitizer on kernel "chemv_kernel(unsigned char*, unsigned char*, unsigned char*, ChemvTilingData)"
[mssanitizer] Sanitizer finished on kernel "chemv_kernel(unsigned char*, unsigned char*, unsigned char*, ChemvTilingData)". No error detected.
[  PASSED  ] 1 test.
```

## upper_reg_large

```text
[mssanitizer] Start memcheck sanitizer on kernel "chemv_kernel(unsigned char*, unsigned char*, unsigned char*, ChemvTilingData)"
[mssanitizer] Sanitizer finished on kernel "chemv_kernel(unsigned char*, unsigned char*, unsigned char*, ChemvTilingData)". No error detected.
[  PASSED  ] 1 test.
```

## lower_reg_large

```text
[mssanitizer] Start memcheck sanitizer on kernel "chemv_kernel(unsigned char*, unsigned char*, unsigned char*, ChemvTilingData)"
[mssanitizer] Sanitizer finished on kernel "chemv_kernel(unsigned char*, unsigned char*, unsigned char*, ChemvTilingData)". No error detected.
[  PASSED  ] 1 test.
```

## positive_stride_padding

```text
[mssanitizer] Start memcheck sanitizer on kernel "chemv_kernel(unsigned char*, unsigned char*, unsigned char*, ChemvTilingData)"
[mssanitizer] Sanitizer finished on kernel "chemv_kernel(unsigned char*, unsigned char*, unsigned char*, ChemvTilingData)". No error detected.
[  PASSED  ] 1 test.
```

## reg_padding

```text
[mssanitizer] Start memcheck sanitizer on kernel "chemv_kernel(unsigned char*, unsigned char*, unsigned char*, ChemvTilingData)"
[mssanitizer] Sanitizer finished on kernel "chemv_kernel(unsigned char*, unsigned char*, unsigned char*, ChemvTilingData)". No error detected.
[  PASSED  ] 1 test.
```

## negative_stride

```text
[mssanitizer] Start memcheck sanitizer on kernel "chemv_kernel(unsigned char*, unsigned char*, unsigned char*, ChemvTilingData)"
[mssanitizer] Sanitizer finished on kernel "chemv_kernel(unsigned char*, unsigned char*, unsigned char*, ChemvTilingData)". No error detected.
[  PASSED  ] 1 test.
```

## alpha_zero_beta_zero

```text
[mssanitizer] Start memcheck sanitizer on kernel "chemv_kernel(unsigned char*, unsigned char*, unsigned char*, ChemvTilingData)"
[mssanitizer] Sanitizer finished on kernel "chemv_kernel(unsigned char*, unsigned char*, unsigned char*, ChemvTilingData)". No error detected.
[  PASSED  ] 1 test.
```

## alpha_zero_beta_other

```text
[mssanitizer] Start memcheck sanitizer on kernel "chemv_kernel(unsigned char*, unsigned char*, unsigned char*, ChemvTilingData)"
[mssanitizer] Sanitizer finished on kernel "chemv_kernel(unsigned char*, unsigned char*, unsigned char*, ChemvTilingData)". No error detected.
[  PASSED  ] 1 test.
```

## beta_one_compute

```text
[mssanitizer] Start memcheck sanitizer on kernel "chemv_kernel(unsigned char*, unsigned char*, unsigned char*, ChemvTilingData)"
[mssanitizer] Sanitizer finished on kernel "chemv_kernel(unsigned char*, unsigned char*, unsigned char*, ChemvTilingData)". No error detected.
[  PASSED  ] 1 test.
```

## upper_ex_tail

```text
[mssanitizer] Start memcheck sanitizer on kernel "chemv_kernel(unsigned char*, unsigned char*, unsigned char*, ChemvTilingData)"
[mssanitizer] Sanitizer finished on kernel "chemv_kernel(unsigned char*, unsigned char*, unsigned char*, ChemvTilingData)". No error detected.
[  PASSED  ] 1 test.
```

## lower_ex_tail

```text
[mssanitizer] Start memcheck sanitizer on kernel "chemv_kernel(unsigned char*, unsigned char*, unsigned char*, ChemvTilingData)"
[mssanitizer] Sanitizer finished on kernel "chemv_kernel(unsigned char*, unsigned char*, unsigned char*, ChemvTilingData)". No error detected.
[  PASSED  ] 1 test.
```

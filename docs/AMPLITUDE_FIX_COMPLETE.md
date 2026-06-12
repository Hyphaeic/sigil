# Amplitude Normalization Remediation — Complete ✅

**Date:** June 11, 2026
**Scope:** Fix the amplitude normalization inconsistency across the impulse/pulse/convolution pipeline, plus two adjacent correctness bugs it exposed
**Status:** **COMPLETE** (all fixes implemented, tested, and verified end-to-end)

---

## Executive Summary

The pipeline mixed discrete-time and continuous-time impulse-response conventions, so every end-to-end run produced physically meaningless amplitudes. Statistical eye levels came out at ~10⁻¹² V ("FAIL - Eye is closed" in every committed reference output in `examples/output*/`), and bit-by-bit amplitudes depended on the resampling ratio. Because a closed eye is plausible for a lossy channel at 32 GT/s, the symptom was never questioned.

- ✅ **CRIT-NEW-005**: Impulse amplitude units inconsistency (statistical eye ~10⁻¹² V)
- ✅ **CRIT-NEW-006**: Sequential overlap-save convolution returned all zeros
- ✅ **HIGH-NEW-007**: Statistical mode collapsed to 1 sample/UI (HIGH-NEW-003 revision)
- ✅ **HIGH-NEW-008**: `StatisticalEye::eye_height()` measured at the crossing region — every eye reported closed

**Test results:** 94/95 unit tests pass (1 pre-existing `lib-ibis` failure unrelated to these fixes; 5 new regression tests added).

**End-to-end verification (synthetic test channel, PCIe Gen 5):**

| Quantity | Before | After |
|----------|--------|-------|
| Pulse response peak | ~5×10⁻¹¹ V | 0.70 V |
| Statistical eye height | −1.4×10⁻¹² V (closed) | 0.75 V, 0.67 UI (open) |
| Differential (SDD21) eye height | closed | 0.72 V, 0.67 UI (open) |
| Bit-by-bit output waveform peak | dt-ratio dependent | 1.14 V (±1 V NRZ + overshoot) |

---

## Issue 1: CRIT-NEW-005 — Impulse Amplitude Units Inconsistency

### Problem

Three components disagreed about the units of the impulse-response `Waveform`:

1. `sparam_to_impulse` (lib-dsp/src/sparam_convert.rs) returned the raw 1/N-normalized IFFT — dimensionless discrete h[n] with Σh[n] = S21(0) ≈ 1.
2. `impulse_to_pulse` integrated with `cumsum += sample * dt` (the CRIT-PHYS-003 fix), which assumes h(t) in units of 1/s. With the dimensionless input, pulse amplitudes came out ≈ dt ≈ 2.5×10⁻¹¹ instead of ≈ 1 V. The unit test passed only because it hand-built an impulse with samples = 1/dt.
3. `ConvolutionEngine` performed plain discrete convolution (no dt), correct for dimensionless taps at the native dt — but the orchestrator resamples the impulse (25 ps → 0.488 ps for 64 samples/UI) with value-preserving Lanczos interpolation, which inflates Σh[n] (and therefore output amplitudes) by the resampling ratio (~51×).

### Fix Implemented

One convention everywhere: **a `Waveform` holding an impulse response contains point samples of the continuous-time h(t), in units of 1/s.** Any discrete sum approximating an integral must carry ×dt.

1. **`sparam_to_impulse`** now scales the IFFT output by 1/dt (equivalently N·df), implementing the inverse Fourier integral h(t) = ∫H(f)e^{j2πft}df ≈ df·ΣH[k]e^{...}. Invariant: Σh[n]·dt ≈ S21(0).
2. **`impulse_to_pulse`** unchanged — its ×dt integration is now dimensionally consistent.
3. **`ConvolutionEngine::from_waveform[_with_strategy]`** converts h(t) to discrete taps h[n]·dt at construction, so discrete convolution approximates the convolution integral. Raw slice constructors (`new`/`with_strategy`) keep plain discrete semantics for generic DSP use.
4. `convolve_waveform*` now warn if the input waveform's dt differs from the impulse's (a units error; amplitudes would scale by the dt ratio).

This makes all amplitudes invariant to FFT size and resampling density: value-preserving resampling is now *correct* for impulse waveforms, because the ×dt is applied with the post-resampling dt.

**Files:** `crates/lib-dsp/src/sparam_convert.rs`, `crates/lib-dsp/src/convolution.rs`

**Tests:** `test_impulse_integral_equals_dc_response`, `test_pulse_amplitude_is_physical` (FFT-size invariance), `test_from_waveform_physical_scaling`, `test_from_waveform_dt_invariance`

---

## Issue 2: CRIT-NEW-006 — Sequential Overlap-Save Returned All Zeros

### Problem

`convolve_sequential` (taken whenever the input fits in ≤2 chunks) iterated with:

```rust
let mut input_pos: isize = -(self.overlap as isize);
while (input_pos as usize) < input.len() + self.overlap { ... }
```

The initial negative `input_pos` wraps to a huge value in the `as usize` cast, so the loop body never executed and the output was identically zero for **any** small input. Existing tests masked this: the long-signal test took the parallel path, and the steady-state tests only compared the (zero) output against itself.

Additionally, both paths computed the chunk count from the input length rather than the output length, dropping up to `impulse_len − 1` tail samples.

### Fix Implemented

The sequential path now uses the same per-chunk indexing as the parallel path (signed `src_idx` with explicit bounds checks), and both paths compute `num_chunks = ceil(output_len / valid_size)` so the convolution tail is produced.

**Files:** `crates/lib-dsp/src/convolution.rs`

**Tests:** `test_convolve_small_input_matches_direct` (every sample, including tail, checked against `direct_convolve`)

---

## Issue 3: HIGH-NEW-007 — Statistical Mode Collapsed to 1 Sample/UI

### Problem

The HIGH-NEW-003 fix derived `samples_per_ui` from the waveform dt (`round(UI/dt)`). The pulse from `sparam_to_pulse` is sampled on the FFT grid — for the synthetic channel dt = 25 ps vs UI = 31.25 ps — so statistical analysis ran at **1 sample/UI**, destroying all phase resolution (eye width degenerated to 0).

### Fix Implemented

`run_statistical` now resamples the pulse onto the UI-aligned grid at the configured `samples_per_ui` (mirroring what `run_bit_by_bit` already did for the impulse), satisfying IBIS 7.2 §11.2 by construction instead of degrading the analysis. Value-preserving resampling is correct here (the pulse is a sampled continuous voltage response).

**Files:** `crates/kernel-cli/src/orchestrator.rs`

---

## Issue 4: HIGH-NEW-008 — Eye Height Measured at the Crossing Region

### Problem

`StatisticalEye::eye_height()` returned the **minimum** opening across all phases. The minimum always lands in the UI crossing region, which is closed for any real channel — so every simulation reported a negative eye height and "FAIL - Eye is closed", regardless of actual margins. (Flagged as a "pre-existing calculation issue" in `eye.rs` test comments; it also made CRIT-NEW-005 invisible.)

### Fix Implemented

`eye_height()` now returns the **maximum** opening across phases — the opening at the optimal sampling phase, which is where a receiver samples. Returns 0.0 when the eye is closed at every phase.

**Files:** `crates/lib-types/src/waveform.rs`

---

## Not Addressed (Pre-Existing, Out of Scope)

- **Bit-by-bit `EyeAnalyzer::compute_metrics`** hardcodes the eye center at phase `samples_per_ui/2` and scans for density in a fixed center voltage band. Without phase alignment to the actual eye center, its width metric is unreliable (reported 0.02 UI on a clearly open eye). Amplitudes feeding it are now correct; the metric itself needs an optimal-phase search like the statistical analyzer.
- **`examples/simple_channel.rs`** is stale demo code (not a cargo target, missing newer `ConversionConfig` fields, and convolves with the pulse rather than the impulse).
- The `lib-ibis` `test_parse_sample_ibs` failure (`[IBIS Ver]` same-line value parsing) predates and is unrelated to this work.

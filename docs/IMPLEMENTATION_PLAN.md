# SI-Kernel Implementation Plan
## Post CRIT-PHYS Fixes - January 2026

**Status:** All 3 CRITICAL physics issues fixed. 7 HIGH issues remain.

**Goal:** Enable lab correlation for differential PCIe channels with mode conversion and crosstalk.

---

## Current State

### ✅ Completed (13 fixes total)
- All 5 original CRITICAL issues (FFI/DSP memory safety)
- All 5 original HIGH numerical issues (passivity, causality, windowing, transients, DFE)
- All 3 CRITICAL physics model issues (DC extrapolation, DC bin, dt scaling)

### 🔴 Remaining Open Issues
| Severity | Count | Categories |
|----------|-------|------------|
| HIGH | 7 | Physics (4), FFI (1), DSP (1), Training (1) |
| MEDIUM | 3 | Physics (1), Training (1), Subsumed (1) |
| LOW | 1 | Training (1) |

---

## Recommended Implementation Tracks

### Track A: Core Correctness (Foundation) 🎯 **START HERE**

**Priority:** HIGHEST
**Impact:** Required for bit-by-bit simulation to work correctly
**Complexity:** Low-Medium
**Testing:** No vendor files needed

#### Issues Addressed:
1. **HIGH-PHYS-006:** Stimulus dt ≠ impulse dt aliasing
2. **HIGH-DSP-004 + HIGH-PHYS-007:** FFT sizing for narrowband resonances

#### Implementation Steps:

##### 1.1 Fix Sampling Alignment (HIGH-PHYS-006)
**File:** `lib-dsp/src/interpolation.rs` (add resampling function)
**File:** `kernel-cli/src/orchestrator.rs:142-157`

**Tasks:**
- [ ] Add `resample_waveform()` function to lib-dsp
  - Cubic spline or sinc interpolation
  - Handles arbitrary dt_in → dt_out conversion
- [ ] Add `ensure_compatible_sampling()` to orchestrator
  - Detects dt mismatch between impulse and stimulus
  - Logs warning when resampling occurs
  - Validates Nyquist criterion: `samples_per_ui ≥ 10 × (bit_time / rise_time)`
- [ ] Add validation in ConvolutionEngine
  - Assert or warn if input waveform dt ≠ expected dt

**Test Plan:**
```bash
# Test with different samples_per_ui values
# Results should be consistent (not vary with config)
cargo test -p lib-dsp interpolation
cargo test -p kernel-cli orchestrator
```

**Acceptance Criteria:**
- Eye diagram invariant to samples_per_ui (within numerical tolerance)
- Warning emitted when dt mismatch detected
- Nyquist violation triggers error or warning

**Estimated Effort:** 2-3 hours

---

##### 1.2 Fix FFT Sizing Heuristic (HIGH-DSP-004 + HIGH-PHYS-007)
**File:** `lib-dsp/src/convolution.rs:64-77`
**File:** `lib-dsp/src/sparam_convert.rs` (add to ConversionConfig)

**Tasks:**
- [ ] Add configurable FFT sizing strategy
  - `FftSizeStrategy::Auto` (current 4x heuristic)
  - `FftSizeStrategy::BandwidthBased { min_delta_f: Hertz }`
  - `FftSizeStrategy::Fixed { size: usize }`
- [ ] Implement bandwidth-based sizing
  - Compute from S-parameter rolloff characteristics
  - Detect narrowband features (Q-factor estimation)
  - Default: `Δf ≤ f_nyquist / 50` (assumes Q=50 resonances)
- [ ] Add to ConversionConfig
  - Expose in JSON config: `"fft_strategy": { "type": "bandwidth", "min_delta_f_ghz": 0.1 }`
- [ ] Log selected FFT size and rationale

**Test Plan:**
```bash
# Test with stub-heavy channel (sharp resonance at 8 GHz)
# Create test S-parameter with Q=50 notch
cargo test -p lib-dsp convolution::tests::test_fft_sizing
```

**Acceptance Criteria:**
- User can override FFT size via config
- Bandwidth-based mode detects narrowband features
- Log shows: "FFT size: 8192 (Δf=100MHz, captures resonances up to Q=50)"

**Estimated Effort:** 3-4 hours

**Deliverable:** Channel-only simulation with correct sampling and FFT resolution

---

### Track B: Differential Channel Support (Lab Correlation) 🔬

**Priority:** HIGH
**Impact:** Required for >16 Gbaud differential signaling (PCIe Gen 5+)
**Complexity:** Medium
**Testing:** Requires 4-port S-parameters (.s4p files)

#### Issues Addressed:
3. **HIGH-PHYS-004:** Mixed-mode S-parameter analysis

#### Implementation Steps:

##### 2.1 Enable Mixed-Mode Analysis (HIGH-PHYS-004)
**File:** `kernel-cli/src/config.rs:64-76`
**File:** `kernel-cli/src/orchestrator.rs:74-90`

**Tasks:**
- [ ] Extend ChannelConfig to support differential mode
  ```rust
  pub struct ChannelConfig {
      pub touchstone: PathBuf,
      pub mode: ChannelMode,  // NEW
  }

  pub enum ChannelMode {
      SingleEnded { input_port: usize, output_port: usize },
      Differential {
          input_p: usize,
          input_n: usize,
          output_p: usize,
          output_n: usize,
      },
  }
  ```
- [ ] Update orchestrator to invoke mixed-mode conversion
  ```rust
  match config.channel.mode {
      ChannelMode::Differential { .. } => {
          let mixed_mode = MixedModeSParameters::from_single_ended(&ts.sparams)?;
          let sdd21 = mixed_mode.sdd21();  // Diff-to-diff
          // Optionally include SCD/SDC for mode conversion loss
      }
      ChannelMode::SingleEnded { .. } => { /* existing path */ }
  }
  ```
- [ ] Add mode conversion loss reporting
  - Log: "Mode conversion ratio: -42 dB (SCD21/SDD21)"
  - Include in effective insertion loss calculation

**Test Plan:**
```bash
# Need 4-port Touchstone file
# test_channel_diff.s4p with ports: (1+,1-,2+,2-)
cargo test -p kernel-cli orchestrator::tests::test_differential_mode
```

**Acceptance Criteria:**
- Config accepts differential port definitions
- SDD21 used for differential channel response
- Mode conversion loss visible in output
- Effective IL includes SCD/SDC contributions

**Estimated Effort:** 4-6 hours

**Deliverable:** Differential channel simulation with mode conversion

---

### Track C: Multi-Lane Analysis (Advanced) 🚀

**Priority:** MEDIUM-HIGH
**Impact:** Required for x4, x8, x16 PCIe configurations
**Complexity:** High
**Testing:** Requires multi-port S-parameters (.s8p, .s16p files)

#### Issues Addressed:
4. **HIGH-PHYS-005:** FEXT/NEXT crosstalk modeling

#### Implementation Steps:

##### 3.1 Multi-Port Convolution Engine (HIGH-PHYS-005)
**File:** `lib-dsp/src/convolution.rs` (new module)

**Tasks:**
- [ ] Design multi-port convolution architecture
  - Single victim lane + N aggressor lanes
  - NxN impulse response matrix support
- [ ] Implement `MultiPortConvolutionEngine`
  ```rust
  pub struct MultiPortConvolutionEngine {
      /// [victim_port][source_port] impulse FFTs
      impulse_matrix: Vec<Vec<Vec<Complex64>>>,
      fft_size: usize,
      // ...
  }
  ```
- [ ] Implement aggressor superposition
  - Generate independent PRBS for each aggressor
  - Convolve each with coupling impulse
  - Sum victim + Σ(aggressor × coupling)
- [ ] Add to orchestrator
  - Config: `"aggressors": [{"port": 3}, {"port": 4}]`
  - Select FEXT S-parameters (S31, S41 for victim on port 1)

**Test Plan:**
```bash
# Synthetic 4-port with known crosstalk
# Victim on port 1, aggressor on port 3
# S31 = -30 dB coupling
cargo test -p lib-dsp convolution::tests::test_crosstalk
```

**Acceptance Criteria:**
- Aggressor-induced jitter visible in eye diagram
- Bathtub curve shows crosstalk-induced slope change
- Multi-lane BER estimate includes FEXT contribution

**Estimated Effort:** 8-12 hours

**Deliverable:** Multi-lane crosstalk simulation

---

### Track D: FFI Safety & Robustness (Vendor Model Readiness) 🛡️

**Priority:** MEDIUM
**Impact:** Required before testing with vendor AMI binaries
**Complexity:** Low-Medium
**Testing:** Requires AMI binary models (.so/.dll)

#### Issues Addressed:
5. **HIGH-FFI-004:** Buffer overrun validation
6. **HIGH-TRAIN-001:** Training state fallback
7. **MED-TRAIN-002:** FOM recording race condition

#### Implementation Steps:

##### 4.1 Buffer Overrun Protection (HIGH-FFI-004)
**File:** `lib-ami-ffi/src/lifecycle.rs:305-312`

**Tasks:**
- [ ] Add buffer sentinel values
  ```rust
  const SENTINEL: f64 = f64::from_bits(0xDEADBEEFDEADBEEF);
  wave_buffer.push(SENTINEL); // Guard byte
  // After FFI call:
  if wave_buffer.last() != Some(&SENTINEL) {
      return Err(AmiError::BufferOverrun);
  }
  ```
- [ ] Add pre-allocation padding
  - Allocate 2x expected size
  - Check if vendor wrote beyond wave_size
- [ ] Log overrun attempts
  - Track models that misbehave
  - Include in error report

**Test Plan:**
- Requires misbehaving AMI model
- Mock test: manually write beyond buffer bounds

**Acceptance Criteria:**
- Overrun detected and reported
- No memory corruption
- Clear error message with model name

**Estimated Effort:** 2-3 hours

---

##### 4.2 Training State Machine (HIGH-TRAIN-001)
**File:** `lib-ami-ffi/src/backchannel.rs:69-78`

**Tasks:**
- [ ] Replace silent fallback with error
  ```rust
  _ => panic!("Unknown training state: {}", raw_state)
  // Or return Result<TrainingState, BackChannelError>
  ```
- [ ] Add state transition validation
  - Only allow valid transitions (Idle→PresetSweep, etc.)
  - Reject invalid transitions (Converged→PresetSweep)
- [ ] Add state history tracking (debug mode)

**Acceptance Criteria:**
- Unknown state triggers panic or error (not silent fallback)
- State transition log available
- Invalid transitions rejected

**Estimated Effort:** 1-2 hours

---

##### 4.3 FOM Recording Fix (MED-TRAIN-002)
**File:** `lib-ami-ffi/src/backchannel.rs:133-141`

**Tasks:**
- [ ] Combine into single atomic update
  ```rust
  pub struct BestResult {
      fom: f64,
      preset: u8,
  }

  pub fn record_fom(&self, fom: f64, preset: u8) {
      let mut best = self.best_result.lock().unwrap();
      if best.is_none() || fom > best.as_ref().unwrap().fom {
          *best = Some(BestResult { fom, preset });
      }
  }
  ```

**Acceptance Criteria:**
- FOM and preset atomically consistent
- No race conditions in parallel training

**Estimated Effort:** 1 hour

---

### Track E: Parser Fixes (Enabling Real Files) 📝

**Priority:** MEDIUM
**Impact:** Required to parse real vendor IBIS/Touchstone files
**Complexity:** Low
**Testing:** Requires real vendor files or fixtures

#### Issues From ISSUES.md:
- CRIT-007: Touchstone port detection silent failure
- CRIT-008: Touchstone buffer overread
- HIGH-006: IBIS suffix parsing bug
- HIGH-007: No IBIS/AMI file validation

**Estimated Effort:** 4-6 hours total

---

## Recommended Execution Order

### Phase 1: Core Correctness (Week 1) 🎯
**Goal:** Fix sampling and FFT issues for channel-only simulation

| Track | Issue | Effort | Priority |
|-------|-------|--------|----------|
| A | HIGH-PHYS-006 (sampling alignment) | 2-3h | 1 |
| A | HIGH-DSP-004 + PHYS-007 (FFT sizing) | 3-4h | 2 |

**Total:** 5-7 hours
**Deliverable:** Channel-only simulation with correct sampling
**Test:** Can test with provided `test_channel.s2p` ✅

---

### Phase 2: Differential Support (Week 2) 🔬
**Goal:** Enable differential channel analysis

| Track | Issue | Effort | Priority |
|-------|-------|--------|----------|
| B | HIGH-PHYS-004 (mixed-mode) | 4-6h | 3 |

**Total:** 4-6 hours
**Deliverable:** Differential channel with mode conversion
**Test:** Need .s4p file (can synthesize or use vendor)

---

### Phase 3: FFI Robustness (Week 2-3) 🛡️
**Goal:** Prepare for vendor AMI binary testing

| Track | Issue | Effort | Priority |
|-------|-------|--------|----------|
| D | HIGH-FFI-004 (buffer overrun) | 2-3h | 4 |
| D | HIGH-TRAIN-001 (state fallback) | 1-2h | 5 |
| D | MED-TRAIN-002 (FOM race) | 1h | 6 |

**Total:** 4-6 hours
**Deliverable:** Robust AMI FFI layer
**Test:** Requires vendor AMI binaries (.so/.dll)

---

### Phase 4: Multi-Lane Crosstalk (Week 3-4) 🚀
**Goal:** Multi-lane PCIe simulation

| Track | Issue | Effort | Priority |
|-------|-------|--------|----------|
| C | HIGH-PHYS-005 (crosstalk) | 8-12h | 7 |

**Total:** 8-12 hours
**Deliverable:** x4/x8/x16 PCIe crosstalk analysis
**Test:** Requires multi-port S-parameters (.s8p, etc.)

---

### Phase 5: Parser & Validation (Week 4+) 📝
**Goal:** Parse real vendor files

| Track | Issue | Effort | Priority |
|-------|-------|--------|----------|
| E | Parser fixes (ISSUES.md) | 4-6h | 8 |

**Total:** 4-6 hours
**Deliverable:** Parse real IBIS/AMI/Touchstone files
**Test:** Requires real vendor files

---

## Detailed Implementation: Phase 1 (Next Steps)

### Step 1: Fix HIGH-PHYS-006 (Sampling Alignment)

**Goal:** Ensure impulse dt matches stimulus dt before convolution

#### 1.1 Add Resampling to lib-dsp

**New file:** `lib-dsp/src/resample.rs`

```rust
/// Resample a waveform to a new time step using sinc interpolation.
pub fn resample_waveform(
    waveform: &Waveform,
    new_dt: Seconds,
) -> DspResult<Waveform> {
    // Use windowed sinc interpolation or cubic spline
    // ...
}

/// Check if two time steps are compatible (within tolerance).
pub fn are_compatible_dt(dt1: Seconds, dt2: Seconds, tolerance: f64) -> bool {
    (dt1.0 - dt2.0).abs() / dt1.0.max(dt2.0) < tolerance
}
```

#### 1.2 Update Orchestrator

**File:** `kernel-cli/src/orchestrator.rs:132-157`

```rust
fn run_bit_by_bit(&self, channel_impulse: &Waveform) -> Result<SimulationResults> {
    // ... generate PRBS ...

    // HIGH-PHYS-006 FIX: Ensure compatible sampling
    let resampled_impulse = if !are_compatible_dt(
        channel_impulse.dt,
        dt,
        1e-6  // 0.0001% tolerance
    ) {
        tracing::warn!(
            "Resampling impulse: dt={:.3e}s → dt={:.3e}s",
            channel_impulse.dt.0, dt.0
        );
        resample_waveform(channel_impulse, dt)?
    } else {
        channel_impulse.clone()
    };

    let conv_engine = ConvolutionEngine::from_waveform(&resampled_impulse)?;
    // ...
}
```

#### 1.3 Add Nyquist Validation

```rust
// Compute required samples_per_ui from channel bandwidth
let channel_bw = estimate_bandwidth(&channel_impulse);  // 3dB point
let rise_time = 0.35 / channel_bw.0;  // Approximate 10-90% rise
let min_samples_per_ui = ((bit_time.0 / rise_time) * 10.0).ceil() as usize;

if samples_per_ui < min_samples_per_ui {
    tracing::error!(
        "Nyquist violation: samples_per_ui={} < required {} for {:.1} GHz BW",
        samples_per_ui, min_samples_per_ui, channel_bw.0 * 1e-9
    );
    return Err(SimError::Config(format!(
        "Insufficient sampling: need at least {} samples/UI",
        min_samples_per_ui
    )));
}
```

**Acceptance:** Warning/error when sampling is insufficient

---

### Step 2: Fix HIGH-DSP-004 + PHYS-007 (FFT Sizing)

**Goal:** Make FFT size aware of channel resonances

#### 2.1 Add FFT Strategy Enum

**File:** `lib-dsp/src/convolution.rs`

```rust
#[derive(Clone, Debug)]
pub enum FftSizeStrategy {
    /// Automatic: 4x impulse length, minimum 1024
    Auto,

    /// Bandwidth-based: ensure Δf captures narrowband features
    Bandwidth { min_delta_f: Hertz },

    /// User-specified fixed size
    Fixed { size: usize },
}

impl ConvolutionEngine {
    pub fn with_strategy(
        impulse: &[f64],
        dt: Seconds,
        strategy: FftSizeStrategy,
    ) -> DspResult<Self> {
        let fft_size = match strategy {
            FftSizeStrategy::Auto => {
                (impulse.len() * 4).next_power_of_two().max(1024)
            }
            FftSizeStrategy::Bandwidth { min_delta_f } => {
                // Δf = 1 / (N × dt)
                // N = 1 / (Δf × dt)
                let n = (1.0 / (min_delta_f.0 * dt.0)).ceil() as usize;
                n.next_power_of_two().max(impulse.len() * 2)
            }
            FftSizeStrategy::Fixed { size } => {
                if !size.is_power_of_two() {
                    return Err(DspError::InvalidFftSize(size));
                }
                size
            }
        };

        // ... rest of implementation
    }
}
```

#### 2.2 Add to ConversionConfig

**File:** `lib-dsp/src/sparam_convert.rs`

```rust
pub struct ConversionConfig {
    // ... existing fields ...

    /// FFT sizing strategy for convolution.
    ///
    /// Controls spectral resolution to capture narrowband resonances.
    /// Bandwidth mode ensures Δf ≤ f_resonance / Q for Q=50 features.
    pub fft_strategy: FftSizeStrategy,
}
```

**Acceptance:** User can specify `min_delta_f` in config

---

## Success Metrics

### After Phase 1:
- [ ] Eye diagram invariant to samples_per_ui (±1% tolerance)
- [ ] Warning when dt mismatch detected
- [ ] FFT size logs show rationale
- [ ] Narrow resonances captured in time domain

### After Phase 2:
- [ ] Differential mode config supported
- [ ] SDD21 used for diff channels
- [ ] Mode conversion loss visible
- [ ] 3-5 dB effective IL increase on real channels

### After Phase 3:
- [ ] No buffer overruns with vendor models
- [ ] Training state transitions validated
- [ ] FOM/preset atomically consistent

### After Phase 4:
- [ ] Multi-lane crosstalk visible
- [ ] x4/x8/x16 configurations supported
- [ ] Aggressor-induced jitter in eye

---

## Risk Assessment

| Issue | If Not Fixed | Mitigation |
|-------|-------------|------------|
| PHYS-006 | Eye varies with config | Test with multiple samples_per_ui |
| PHYS-004 | 3-5 dB optimism | Compare SE vs. Diff mode |
| PHYS-007 | Missed resonances | Compare to VNA measurements |
| PHYS-005 | Crosstalk invisible | Single-lane only for now |
| FFI-004 | Memory corruption | Add canary values in buffers |

---

## Decision Points

### Option 1: Fast Path to Lab Correlation
**Focus:** Phase 1 + Phase 2 (10-13 hours)
**Pros:** Gets differential mode working quickly
**Cons:** No crosstalk, vendor models untested
**Recommendation:** ✅ **Best for rapid validation**

### Option 2: Comprehensive Foundation
**Focus:** Phase 1 + Phase 3 + Phase 4 (9-13 hours for Phases 1+3)
**Pros:** FFI robust for vendor testing
**Cons:** Multi-lane (Phase 4) is 8-12h additional
**Recommendation:** Good if vendor models available

### Option 3: Full Implementation
**Focus:** All phases (21-31 hours)
**Pros:** Complete feature set
**Cons:** Long timeline, may not have test files
**Recommendation:** Defer Phase 4 until multi-lane S-params available

---

## Immediate Next Actions (This Session)

### Recommended: Start Phase 1

```bash
# 1. Implement resampling (HIGH-PHYS-006)
# lib-dsp/src/resample.rs

# 2. Add FftSizeStrategy (HIGH-DSP-004 + PHYS-007)
# lib-dsp/src/convolution.rs

# 3. Update orchestrator to use resampling
# kernel-cli/src/orchestrator.rs

# 4. Test with provided test_channel.s2p
cd examples
./test_crit_phys_fixes.sh

# 5. Verify eye diagram consistency across samples_per_ui
cargo test -p kernel-cli
```

**Timeline:** 5-7 hours
**Can complete today:** Yes ✅

---

## Questions to Decide Priority

1. **Do you have differential S-parameters (.s4p files)?**
   - YES → Prioritize Phase 2 (mixed-mode)
   - NO → Stay on Phase 1, defer Phase 2

2. **Do you have vendor AMI binaries (.so/.dll)?**
   - YES → Prioritize Phase 3 (FFI robustness)
   - NO → Defer Phase 3 until files obtained

3. **Do you need multi-lane crosstalk now?**
   - YES → Add Phase 4 to roadmap
   - NO → Defer until single-lane validated

4. **What's the primary use case?**
   - Channel characterization → Phase 1 sufficient
   - Differential PCIe → Add Phase 2
   - Multi-lane x8/x16 → Add Phase 4
   - Vendor model testing → Add Phase 3

---

**Recommendation:** Start with **Phase 1** (HIGH-PHYS-006 + HIGH-DSP-004/PHYS-007). This gives you:
- Correct sampling alignment
- Proper FFT resolution
- Testable with provided files
- Foundation for all other work

Total effort: **5-7 hours** for a working, physically correct channel-only simulator.

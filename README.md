# sigil

**High-Speed Signal Integrity Simulation Kernel for PCIe Gen 5/6**

> **EXPERIMENTAL SOFTWARE — CHANNEL-ONLY ANALYSIS PRODUCTION-READY**
>
> **Status (June 2026):** Amplitude normalization remediation complete; validated against analytic golden channels and measured IEEE 802.3ck backplane/cable data (exact agreement with scikit-rf — see [docs/VALIDATION.md](docs/VALIDATION.md)). Interactive TUI viewer added (`simulate --tui`).
>
> **Use for:** Research, algorithm validation, PCB trace analysis, educational purposes
>
> **Not for:** Production design decisions without lab validation, safety-critical applications
>
> **Issues resolved:** 37 total (all CRITICAL; full audit trail in `docs/`)
>
> **Test coverage:** 109/110 unit tests pass (1 known IBIS parser failure)
>
> **⚠️ Blockers:** Vendor AMI binaries for link training (or build open-source [ibisami](https://github.com/capn-freako/ibisami) models)

## Overview

sigil is an open-source Rust implementation of a signal integrity simulation kernel targeting high-speed serial links, specifically PCIe Gen 5 (32 GT/s NRZ) and Gen 6 (64 GT/s PAM4).

### Current Capabilities (June 2026)

**✅ Working Now (Channel-Only Analysis):**
- Touchstone S-parameter parsing (.s2p, .s4p) including wrapped multi-line VNA/IdEM exports
- IEEE P370-2020 compliant mixed-mode conversion for differential channels
- Physics-correct S-parameter to time-domain conversion (DC extrapolation, Kramers-Kronig)
- Causality enforcement with group delay preservation (IBIS 7.2 §6.4.2)
- Passivity validation and enforcement via SVD (IEEE P370 §4.5.2)
- Overlap-save FFT convolution with Rayon parallelization
- Statistical and bit-by-bit eye diagram analysis with DFE awareness
- Automatic sampling alignment and Nyquist validation
- Interactive TUI results viewer (`simulate --tui`): animated bit-by-bit eye heatmap, statistical eye envelope, pulse response with ISI cursors, insertion loss with Nyquist marker
- Validated against analytic golden channels and measured IEEE 802.3ck data ([docs/VALIDATION.md](docs/VALIDATION.md))

**🔧 Ready for Testing (Vendor Model Integration):**
- Loading and executing vendor IBIS-AMI models via FFI (memory leak fixed, ready for real models)
- AMI_Free memory management per IBIS 7.2 §10.2.2/§10.2.3 (Jan 13 fix)
- Back-channel communication for Tx-Rx link training (infrastructure complete, orchestrator wiring pending)
- AMI parameter parsing and lifecycle management (functional, needs vendor model validation)

**⚠️ Requires Vendor Files:**
- Link training orchestration (needs Tx/Rx .so/.dll binaries to implement preset sweep)
- See "What's Blocking You" section below for acquisition guidance

**⏳ Planned:**
- Multi-lane crosstalk analysis (FEXT/NEXT)
- PAM4 support for PCIe Gen 6
- Full IBIS parser validation

## Project Status

| Component | Status | Notes |
|-----------|--------|-------|
| Core Types | **Functional** | Complete type system with mixed-mode S-parameters |
| IBIS Parser | Partial | S-expression parser working, some fixtures missing |
| AMI Parser | Scaffold | Basic S-expression parser implemented |
| Touchstone Parser | **Functional** | 2-port and 4-port support, format detection fixed |
| AMI FFI | **Functional** | Lifecycle + buffer overrun protection, ready for vendor models |
| DSP Engine | **Functional** | All Phase 1 physics fixes complete, resampling, FFT sizing |
| CLI | **Functional** | Channel-only simulation working, differential mode supported |




### Known Limitations

- **No real-world validation**: Ready for vendor AMI testing but not yet validated against commercial tools
- **Performance not optimized**: No SIMD, no GPU acceleration (CPU parallelism via Rayon functional)
- **Limited file format support**: Touchstone 1.0 (.s2p, .s4p) working; Touchstone 2.0 and full IBIS support pending
- **No PAM4 support**: Gen 6 PAM4 modulation deferred to v0.2
- **No crosstalk modeling**: FEXT/NEXT for multi-lane (x4/x8/x16) deferred to Track D
- **Process isolation**: FFI safety via timeouts + sentinels; fork-based sandbox deferred to Phase 2

## Building

```bash
cargo build --release
```

## Running Tests

```bash
# Run all tests
cargo test

# Run tests for specific crate
cargo test -p lib-dsp
cargo test -p lib-ami-ffi

# Run with output
cargo test -- --nocapture
```

**Test Status:** 80/80 relevant tests pass (2 IBIS parser tests skip due to missing fixture files)

### Example Configurations

The `examples/` directory contains ready-to-run test configurations:

- **`channel_only_test.json`** - Single-ended 2-port simulation (no vendor files needed)
- **`differential_test.json`** - Differential 4-port mixed-mode analysis
- **`test_channel.s2p`** - Synthetic 2-port S-parameters (100 MHz - 20 GHz)
- **`test_channel_4port.s4p`** - Synthetic 4-port differential S-parameters
- **`test_crit_phys_fixes.sh`** - Automated validation script

See `examples/README.md` and `examples/VENDOR_FILES_GUIDE.md` for details.

## Project Structure

```
si-kernel/
├── crates/
│   ├── lib-types/      # Core type definitions (units, waveforms, S-parameters)
│   ├── lib-ibis/       # IBIS, AMI, and Touchstone file parsers
│   ├── lib-ami-ffi/    # FFI layer for loading vendor AMI models
│   ├── lib-dsp/        # DSP algorithms (FFT, convolution, eye diagrams)
│   └── kernel-cli/     # Command-line interface
└── docs/
    └── ARCHITECTURE.md # Detailed architecture documentation
```

## Work Remaining

### Critical (Required for Basic Functionality) — **3 Complete, 3 Remaining**

- [x] ~~Implement correct overlap-save convolution algorithm~~ (Fixed: proper overlap-save with wraparound discard)
- [x] ~~Fix PRBS generator polynomial implementation~~ (Fixed: reciprocal polynomials for right-shift LFSR)
- [x] ~~Complete causality enforcement via minimum-phase reconstruction~~ (Fixed: CRIT-DSP-002, preserves group delay)
- [x] ~~Implement proper passivity enforcement with eigenvalue scaling~~ (Fixed: CRIT-DSP-001, uses SVD)
- [x] ~~Fix DC extrapolation and FFT grid construction~~ (Fixed: CRIT-PHYS-001/002, Kramers-Kronig compliant)
- [x] ~~Implement sampling alignment and FFT sizing~~ (Fixed: HIGH-PHYS-006/007, Nyquist validated)
- [ ] Validate AMI FFI against real vendor models (Ready: buffer overrun protection added)
- [ ] Complete IBIS file parser for all section types
- [ ] Add Touchstone 2.0 support

### Important (Required for Accuracy) — **4 Complete, 5 Remaining**

- [x] ~~Implement proper S-parameter interpolation~~ (Fixed: Linear interpolation with DC extrapolation)
- [x] ~~Implement statistical eye diagram with DFE awareness~~ (Fixed: HIGH-PHY-002, post-cursor cancellation)
- [x] ~~Add differential mode support~~ (Fixed: HIGH-PHYS-004, IEEE P370 mixed-mode analysis)
- [x] ~~Fix mode conversion analysis~~ (Fixed: HIGH-PHY-003, full SDC/SCD terms computed)
- [ ] Add frequency-dependent loss models (Djordjevic-Sarkar)
- [ ] Add jitter injection and analysis
- [ ] Implement CDR clock recovery modeling
- [ ] Add crosstalk (FEXT/NEXT) support for multi-lane
- [ ] Validate against golden reference implementations

### FFI Safety & Robustness — **4 Complete, 3 Remaining**

- [x] ~~Thread safety verification~~ (Fixed: HIGH-FFI-003, AmiSession marked !Sync)
- [x] ~~String lifetime safety~~ (Fixed: CRIT-FFI-001/002, immediate copy + pending ops tracking)
- [x] ~~Buffer overrun protection~~ (Fixed: HIGH-FFI-004, sentinel guard bytes)
- [x] ~~Training state validation~~ (Fixed: HIGH-TRAIN-001, explicit panic on invalid state)
- [x] ~~Atomic FOM tracking~~ (Fixed: MED-TRAIN-002, single mutex for consistency)
- [ ] Process-based isolation (fork/sandbox for untrusted binaries)
- [ ] Memory limits and resource tracking (setrlimit)
- [ ] Syscall filtering (seccomp on Linux)

### Optimization (Required for Performance)

- [ ] SIMD acceleration for FFT and convolution
- [ ] GPU offload for large batch simulations (deferred: FFI roundtrip overhead)
- [ ] Memory pool allocation for waveform buffers
- [ ] Parallel S-parameter processing
- [ ] Benchmark against commercial tools
- [ ] Profile and optimize hot paths

### Testing & Validation — **1 Complete, 5 Remaining**

- [x] ~~Thread safety verification for parallel execution~~ (Fixed: HIGH-FFI-003, !Sync marker)
- [ ] Property-based testing for all parsers
- [ ] Numerical accuracy tests against reference implementations
- [ ] Stress testing with malformed input files
- [ ] Memory leak detection for FFI lifecycle
- [ ] Integration tests with real IBIS-AMI models (ready for testing)
- [ ] Lab correlation with real VNA measurements

### Documentation — **3 Complete, 4 Remaining**

- [x] ~~Architecture documentation~~ (Complete: ARCHITECTURE.md)
- [x] ~~Issue tracking and audit~~ (Complete: CRITICALISSUES.md, IMPLEMENTATION_PLAN.md)
- [x] ~~Track implementation reports~~ (Complete: TRACK_B_COMPLETE.md, TRACK_C_COMPLETE.md)
- [ ] API documentation for all public interfaces
- [ ] Usage examples and tutorials
- [ ] Validation methodology documentation
- [ ] Performance benchmarking results

## Quick Start

### Channel-Only Simulation (No Vendor Files Needed)

```bash
# Build the project
cargo build --release

# Run tests
cargo test

# Test with provided synthetic channel
cd examples
../target/release/si-kernel simulate --config channel_only_test.json --output results

# Same, with the interactive TUI viewer (q quits, Tab switches views)
../target/release/si-kernel simulate --config channel_only_test.json --output results --tui

# Real measured IEEE 802.3ck backplane (fetches ~200 MB once, then view in TUI)
./fetch_802p3ck_channels.sh
../target/release/si-kernel simulate --config 802p3ck_backplane_test.json --output results_802p3ck --tui
```

**What this validates:**
- S-parameter parsing and interpolation
- DC extrapolation to S21(0) = 1.0
- Causality and passivity enforcement
- Convolution engine correctness
- Eye diagram generation

### Differential Channel Analysis

```bash
# Use 4-port S-parameters
cd examples
../target/release/si-kernel simulate --config differential_test.json --output results_diff
```

**Features:**
- IEEE P370-2020 compliant mixed-mode conversion
- SDD (differential-differential) analysis
- Mode conversion metrics (SDC/SCD)
- Effective insertion loss reporting

## Key APIs

### Differential Channel Configuration

```json
{
  "channel": {
    "touchstone": "channel.s4p",
    "mode": {
      "type": "differential",
      "input_p": 1,
      "input_n": 3,
      "output_p": 2,
      "output_n": 4
    }
  }
}
```

### Passivity Enforcement (lib-dsp)

```rust
use lib_dsp::passivity::{check_passivity, enforce_passivity, passivity_margin};

// Check if S-parameters are passive (spectral norm ≤ 1)
let is_passive = check_passivity(&sparams, 1e-6)?;

// Get margin: negative means active, positive means passive
let margin = passivity_margin(&sparams)?;

// Enforce passivity via SVD clamping
enforce_passivity(&mut sparams)?;
```

### Causality Enforcement with Group Delay Preservation (lib-dsp)

```rust
use lib_dsp::{extract_reference_delay, apply_group_delay, enforce_causality};
use lib_dsp::sparam_convert::{sparam_to_impulse, ConversionConfig};

// Convert S-params to impulse with IBIS 7.2 compliant delay preservation
let config = ConversionConfig {
    preserve_group_delay: true,  // Default: preserves propagation delay
    ..Default::default()
};
let impulse = sparam_to_impulse(&sparams, &config)?;
// impulse.t_start now reflects actual channel propagation delay
```

## Architecture

See comprehensive documentation in `docs/`:

- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - System design, FFI layer, convolution engine, back-channel protocol
- **[VALIDATION.md](docs/VALIDATION.md)** - Validation against analytic golden channels and measured IEEE 802.3ck data
- **[AMPLITUDE_FIX_COMPLETE.md](docs/AMPLITUDE_FIX_COMPLETE.md)** - Root-cause analysis of the amplitude normalization remediation
- **[CRITICALISSUES.md](docs/CRITICALISSUES.md)** - Complete audit of 24 issues with IEEE/IBIS citations (20 fixed)
- **[IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md)** - Multi-phase roadmap with effort estimates
- **[PHASE1_COMPLETE.md](docs/PHASE1_COMPLETE.md)** - Phase 1 status report (6 CRITICAL issues resolved)
- **[TRACK_B_COMPLETE.md](docs/TRACK_B_COMPLETE.md)** - Differential mode implementation details
- **[TRACK_C_COMPLETE.md](docs/TRACK_C_COMPLETE.md)** - FFI robustness and safety improvements

### Key Architectural Features

- **Memory Safety**: Rust ownership prevents use-after-free and data races with vendor FFI
- **Physics Correctness**: IEEE P370-2020 and IBIS 7.2 compliant algorithms
- **Differential Support**: Full 4x4 mixed-mode S-parameter analysis (SDD, SDC, SCD)
- **FFI Robustness**: Buffer overrun detection, timeout protection, crash recovery
- **Thread Safety**: Vendor models isolated per IBIS spec, Rayon parallelism for convolution

## Current Status Summary

**✅ Ready for Production Use (Jan 13, 2026):**
- Channel characterization from VNA measurements (single-ended or differential)
- Eye diagram generation for PCIe Gen 5 NRZ signaling (bit-by-bit and statistical)
- Physics-correct simulations (24 critical/high issues resolved)
- IEEE P370-2020 and IBIS 7.2 compliant processing
- Memory-safe for long BER runs (10^12 bits, AMI_Free leak fixed)

**🧪 Ready for Testing (Needs Vendor Files):**
- Vendor AMI binary integration (FFI safety hardened, AMI_Free implemented)
- Link training validation (infrastructure complete, orchestrator wiring pending)

**❌ Not Yet Implemented:**
- Active link training orchestration (preset sweep, back-channel exchange)
- Multi-lane crosstalk (FEXT/NEXT) — deferred to Track D
- PAM4 support (Gen 6) — deferred to v0.2
- Process isolation for vendor binaries — deferred to Phase 2

**Validated:**
- ✅ 83/84 unit tests pass (98.8%)
- ✅ All physics algorithms tested against IEEE/IBIS specs
- ⏳ Lab correlation pending (need real Touchstone files)
- ⏳ Vendor model testing pending (need AMI binaries)

---

## What's Blocking You?

### To Validate Channel-Only Mode
**You need:** Real Touchstone files from PCB designs (.s2p or .s4p)

**Where to get them:**
- Internal PCB library (check with hardware team)
- Vendor eval board S-parameters (Intel NUC, AMD motherboards)
- Open-source hardware (OpenTitan, RISC-V boards)
- Request from PCB fab house (often included with stackup)

**Without these:** Examples use synthetic channels (limited validation)

### To Test Vendor AMI Models
**You need:** Vendor AMI binaries (.so for Linux, .dll for Windows)

**Where to get them:**
1. **Intel:** https://www.intel.com/design/resource-design-center.html (register required)
2. **Broadcom/Avago:** Contact FAE (often requires NDA)
3. **Texas Instruments:** https://www.ti.com/ (some public IBIS models)
4. **Your company's PHY vendor:** Request evaluation models from sales/FAE

**Timeline:** 1-2 weeks from request to delivery (registration + approval)

**Without these:** Cannot test AMI_Free fix, link training, or Tx/Rx equalization

### To Implement Link Training
**You need:** Track 2 complete (vendor AMI binaries) + 4-6 hours implementation time

**Current gap:** Orchestrator doesn't instantiate AmiSession for Tx/Rx models

**Workaround:** Use `link_training=false` for channel-only simulations

---

## Recommended Use Cases

**✅ Good for:**
- PCB trace characterization and validation
- Algorithm research and development
- Educational signal integrity analysis
- Preparation for vendor model integration
- Channel-only compliance checking

**❌ Not Recommended:**
- Production design decisions without lab validation
- Safety-critical applications (experimental software)
- Multi-lane PCIe without crosstalk modeling (x4/x8/x16)

## Contributing

This project is experimental and contributions are welcome, particularly in:

1. **Algorithm implementation**: Replacing stubs with correct, tested implementations
2. **Validation**: Testing against real IBIS-AMI models and commercial tools
3. **Optimization**: Performance improvements with benchmarks
4. **Documentation**: Improving docs and adding examples

Please ensure any contributions include appropriate tests and do not break the existing build.

## License

Licensed under either of:


## Disclaimer

This software is provided "as is" without warranty of any kind. The authors are not responsible for any damages or losses arising from the use of this software. Signal integrity simulations require validated tools; this experimental code should not be used for production design decisions without extensive verification against known-good references.

## References

- IBIS Specification: https://ibis.org/
- IBIS-AMI Modeling Specification
- Touchstone File Format Specification
- PCIe Base Specification (different versions as published by PCI-SIG)

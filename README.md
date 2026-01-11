# sigil

**High-Speed Signal Integrity Simulation Kernel for PCIe Gen 5/6**

> **WARNING: EXPERIMENTAL SOFTWARE**
>
> This project is in early development and is not ready for production use. The codebase requires extensive refinement, optimization, verification, and benchmarking before it can be relied upon for accurate signal integrity analysis.

## Overview

SI-Kernel is an open-source Rust implementation of a signal integrity simulation kernel targeting high-speed serial links, specifically PCIe Gen 5 (32 GT/s NRZ) and Gen 6 (64 GT/s PAM4). It provides infrastructure for:

- Loading and executing vendor IBIS-AMI models via FFI
- Parsing IBIS, AMI, and Touchstone file formats
- Channel response computation via convolution
- S-parameter processing with causality and passivity enforcement
- Back-channel communication for Tx-Rx link training
- Eye diagram generation and analysis

## Project Status

| Component | Status | Notes |
|-----------|--------|-------|
| Core Types | Scaffold | Basic type definitions in place |
| IBIS Parser | Scaffold | Partial implementation, needs validation |
| AMI Parser | Scaffold | S-expression parser implemented |
| Touchstone Parser | Scaffold | Basic 2-port support |
| AMI FFI | Scaffold | Lifecycle management, untested with real models |
| DSP Engine | **Partial** | Passivity and causality enforcement fixed; convolution/PRBS need work |
| CLI | Scaffold | Basic structure only |

### Recent Fixes

The following critical issues have been addressed per IEEE P370-2020 and IBIS 7.2 standards:

| Issue | Description | Status |
|-------|-------------|--------|
| CRIT-DSP-001 | **Passivity Enforcement**: Replaced incorrect element-wise check with SVD-based spectral norm validation per IEEE P370-2020 Section 4.5.2 | Fixed |
| CRIT-DSP-002 | **Causality Group Delay**: Minimum-phase reconstruction now preserves group delay per IBIS 7.2 Section 6.4.2 | Fixed |

### Known Limitations

- **Some DSP algorithms are stubs**: The convolution engine, PRBS generator, and eye diagram code are placeholder implementations that need work
- **No real-world validation**: The code has not been tested against actual IBIS-AMI models or validated against commercial tools
- **Performance not optimized**: No SIMD, no GPU acceleration, naive algorithm implementations
- **Limited file format support**: Only basic Touchstone 1.0 and simple IBIS files
- **No PAM4 support**: Gen 6 PAM4 modulation is not yet implemented

## Building

```bash
cargo build --release
```

## Running Tests

```bash
cargo test
```

Note: Several DSP tests currently fail due to incomplete algorithm implementations.

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

### Critical (Required for Basic Functionality)

- [ ] Implement correct overlap-save convolution algorithm
- [ ] Fix PRBS generator polynomial implementation
- [x] ~~Complete causality enforcement via minimum-phase reconstruction~~ (Fixed: CRIT-DSP-002)
- [x] ~~Implement proper passivity enforcement with eigenvalue scaling~~ (Fixed: CRIT-DSP-001, uses SVD)
- [ ] Validate AMI FFI against real vendor models
- [ ] Complete IBIS file parser for all section types
- [ ] Add Touchstone 2.0 support

### Important (Required for Accuracy)

- [ ] Implement proper S-parameter interpolation (cubic spline)
- [ ] Add frequency-dependent loss models
- [ ] Implement statistical eye diagram with proper PDF accumulation
- [ ] Add jitter injection and analysis
- [ ] Implement CDR clock recovery modeling
- [ ] Add crosstalk (FEXT/NEXT) support
- [ ] Validate against golden reference implementations

### Optimization (Required for Performance)

- [ ] SIMD acceleration for FFT and convolution
- [ ] GPU offload for large batch simulations
- [ ] Memory pool allocation for waveform buffers
- [ ] Parallel S-parameter processing
- [ ] Benchmark against commercial tools
- [ ] Profile and optimize hot paths

### Testing & Validation

- [ ] Property-based testing for all parsers
- [ ] Numerical accuracy tests against reference implementations
- [ ] Stress testing with malformed input files
- [ ] Memory leak detection for FFI lifecycle
- [ ] Thread safety verification for parallel execution
- [ ] Integration tests with real IBIS-AMI models

### Documentation

- [ ] API documentation for all public interfaces
- [ ] Usage examples and tutorials
- [ ] Validation methodology documentation
- [ ] Performance benchmarking results

## Key APIs

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

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed architecture documentation including:

- FFI design for vendor AMI binaries
- Convolution engine design
- S-parameter processing pipeline
- Back-channel training protocol
- Thread safety considerations

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

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
| DSP Engine | Scaffold | Algorithm stubs, several failing tests |
| CLI | Scaffold | Basic structure only |

### Known Limitations

- **DSP algorithms are stubs**: The convolution engine, causality enforcement, PRBS generator, and eye diagram code are placeholder implementations that do not produce correct results
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
- [ ] Complete causality enforcement via minimum-phase reconstruction
- [ ] Implement proper passivity enforcement with eigenvalue scaling
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

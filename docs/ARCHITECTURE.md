# SI-Kernel Architecture Design Document

**Version:** 0.1.0
**Status:** Draft
**Target:** PCIe Gen 5 (32 GT/s NRZ) and Gen 6 (64 GT/s PAM4)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Overview](#2-system-overview)
3. [Workspace Structure](#3-workspace-structure)
4. [Core Data Types](#4-core-data-types)
5. [FFI Design: AMI Binary Interface](#5-ffi-design-ami-binary-interface)
6. [Convolution Engine](#6-convolution-engine)
7. [S-Parameter Processing](#7-s-parameter-processing)
8. [Back-Channel Training Architecture](#8-back-channel-training-architecture)
9. [Simulation Modes](#9-simulation-modes)
10. [Error Handling Strategy](#10-error-handling-strategy)
11. [Performance Considerations](#11-performance-considerations)
12. [Security Model](#12-security-model)
13. [Future Extensions](#13-future-extensions)

---

## 1. Executive Summary

SI-Kernel is an open-source, high-speed IBIS-AMI simulation kernel written in Rust. It provides a standalone "Channel Solver" capable of both statistical and time-domain bit-by-bit simulation for PCIe Gen 5/6 signal integrity analysis.

### Design Principles

1. **Memory Safety First**: Rust's ownership model protects against use-after-free and data races when interfacing with vendor AMI binaries.
2. **Isolation**: Vendor binaries execute in controlled contexts with resource limits and crash recovery.
3. **Performance**: Leverage Rayon for CPU parallelism; defer GPU acceleration to Phase 2.
4. **Standards Compliance**: Full IBIS 7.2 and IBIS-AMI specification support.
5. **Composability**: Each crate has a single responsibility and clear API boundaries.

---

## 2. System Overview

### High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           SI-Kernel Pipeline                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────┐    ┌─────────┐    ┌─────────────┐    ┌─────────────────┐  │
│  │  .ibs   │───▶│ lib-ibis│───▶│  Parsed     │    │                 │  │
│  │  .ami   │    │ Parser  │    │  Model      │    │                 │  │
│  └─────────┘    └─────────┘    └──────┬──────┘    │                 │  │
│                                       │           │                 │  │
│  ┌─────────┐    ┌─────────┐    ┌──────▼──────┐    │   kernel-cli    │  │
│  │  .s4p   │───▶│ lib-dsp │───▶│  Channel    │───▶│   Orchestrator  │  │
│  │  .s8p   │    │ S-Param │    │  Response   │    │                 │  │
│  └─────────┘    └─────────┘    └──────┬──────┘    │                 │  │
│                                       │           │                 │  │
│  ┌─────────┐    ┌─────────┐    ┌──────▼──────┐    │                 │  │
│  │ Tx .dll │───▶│lib-ami- │───▶│  Equalized  │───▶│                 │  │
│  │ Rx .so  │    │  ffi    │    │  Response   │    │                 │  │
│  └─────────┘    └─────────┘    └─────────────┘    └────────┬────────┘  │
│                                                            │           │
│                                      ┌─────────────────────▼────────┐  │
│                                      │     Simulation Results       │  │
│                                      │  • Eye Diagram               │  │
│                                      │  • Bathtub Curve             │  │
│                                      │  • BER Contours              │  │
│                                      └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Simulation Workflow

1. **Extraction**: Convert Touchstone S-parameters to pulse/impulse response via IFFT
2. **Characterization**: Execute `AMI_Init` on Tx/Rx models to obtain equalization responses
3. **Convolution**: Apply channel + Tx + Rx responses to bitstream
4. **Analysis**: Generate eye diagrams, BER estimates, timing margins
5. **Cleanup**: Execute `AMI_Close` to release vendor binary resources

---

## 3. Workspace Structure

```
si-kernel/
├── Cargo.toml                    # Workspace root
├── docs/
│   ├── ARCHITECTURE.md           # This document
│   ├── AMI_FFI_SAFETY.md         # FFI safety analysis
│   └── PHYSICS_NOTES.md          # S-param math derivations
├── crates/
│   ├── lib-types/                # Shared type definitions
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── units.rs          # Physical units (Volts, Seconds, Hz)
│   │       ├── waveform.rs       # Time-domain waveforms
│   │       ├── sparams.rs        # S-parameter matrices
│   │       └── ami.rs            # AMI-specific types
│   │
│   ├── lib-ibis/                 # IBIS/AMI file parser
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── lexer.rs          # Tokenization
│   │       ├── ibs_parser.rs     # .ibs file parser
│   │       ├── ami_parser.rs     # .ami parameter file parser
│   │       ├── touchstone.rs     # .sNp file parser
│   │       └── model.rs          # Parsed model structures
│   │
│   ├── lib-ami-ffi/              # Unsafe FFI boundary
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── loader.rs         # Dynamic library loading
│   │       ├── lifecycle.rs      # Init/GetWave/Close management
│   │       ├── sandbox.rs        # Resource limits and isolation
│   │       ├── error_recovery.rs # Crash handling
│   │       └── backchannel.rs    # Tx-Rx message passing
│   │
│   ├── lib-dsp/                  # Signal processing core
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── fft.rs            # FFT/IFFT wrappers
│   │       ├── convolution.rs    # High-performance convolution
│   │       ├── interpolation.rs  # Frequency interpolation
│   │       ├── causality.rs      # Hilbert transform enforcement
│   │       ├── passivity.rs      # Passivity enforcement
│   │       ├── prbs.rs           # PRBS pattern generation
│   │       └── eye.rs            # Eye diagram computation
│   │
│   └── kernel-cli/               # Main binary
│       └── src/
│           ├── main.rs
│           ├── config.rs         # Simulation configuration
│           ├── orchestrator.rs   # Pipeline coordination
│           └── output.rs         # Result formatting
│
├── examples/
│   ├── simple_channel.rs
│   └── pcie_gen5_full.rs
│
└── tests/
    ├── golden/                   # Reference waveforms
    └── integration/
```

### Crate Dependency Graph

```
                    lib-types
                   /    |    \
                  /     |     \
           lib-ibis  lib-dsp  lib-ami-ffi
                  \     |     /
                   \    |    /
                   kernel-cli
```

---

## 4. Core Data Types

### `lib-types` Foundational Types

```rust
// units.rs - Zero-cost unit wrappers for type safety
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Seconds(pub f64);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hertz(pub f64);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Volts(pub f64);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ohms(pub f64);

// waveform.rs - Time-domain signal representation
pub struct Waveform {
    /// Sample values (voltage or normalized)
    pub samples: Vec<f64>,
    /// Time step between samples
    pub dt: Seconds,
    /// Time of first sample (may be negative for acausal responses)
    pub t_start: Seconds,
}

// sparams.rs - S-parameter data structures
pub struct SParameters {
    /// Frequency points
    pub frequencies: Vec<Hertz>,
    /// S-parameter matrices at each frequency [freq_idx][row][col]
    pub matrices: Vec<SMatrix>,
    /// Reference impedance (typically 50 ohms)
    pub z0: Ohms,
    /// Number of ports
    pub num_ports: usize,
}

#[derive(Clone, Copy)]
pub struct Complex64 {
    pub re: f64,
    pub im: f64,
}

pub type SMatrix = ndarray::Array2<Complex64>;

// ami.rs - AMI model interface types
#[repr(C)]
pub struct AmiWave {
    pub ptr: *mut f64,
    pub len: usize,
    pub clock_times: *mut f64,
}

pub struct AmiParameters {
    pub params: HashMap<String, AmiValue>,
}

pub enum AmiValue {
    Float(f64),
    Integer(i64),
    String(String),
    List(Vec<AmiValue>),
    Table(Vec<Vec<f64>>),
}
```

---

## 5. FFI Design: AMI Binary Interface

### 5.1 The Challenge

Vendor-supplied AMI models (`.dll`/`.so`) are compiled C/C++ binaries with:
- Unknown internal state
- Potential for memory leaks
- Risk of segfaults or infinite loops
- Platform-specific calling conventions

### 5.2 AMI Function Signatures (IBIS-AMI Standard)

```c
// AMI_Init: Initialize model, optionally return impulse response modification
long AMI_Init(
    double *impulse_matrix,      // [in/out] Channel impulse response
    long    row_size,            // Number of samples
    long    aggressors,          // Number of aggressor channels
    double  sample_interval,     // Time step (seconds)
    double  bit_time,            // UI duration (seconds)
    char   *AMI_parameters_in,   // Input parameter string
    char  **AMI_parameters_out,  // Output parameter string
    void  **AMI_memory_handle,   // Opaque handle for model state
    char  **msg                  // Error/info message
);

// AMI_GetWave: Process a waveform through the model
long AMI_GetWave(
    double *wave,                // [in/out] Waveform to process
    long    wave_size,           // Number of samples
    double *clock_times,         // [out] CDR clock positions
    char  **AMI_parameters_out,  // Output parameter string
    void   *AMI_memory_handle    // Handle from AMI_Init
);

// AMI_Close: Release all model resources
long AMI_Close(
    void *AMI_memory_handle      // Handle from AMI_Init
);
```

### 5.3 Safe Wrapper Architecture

```rust
// loader.rs
pub struct AmiLibrary {
    library: libloading::Library,
    ami_init: Symbol<AmiInitFn>,
    ami_getwave: Symbol<AmiGetWaveFn>,
    ami_close: Symbol<AmiCloseFn>,
}

impl AmiLibrary {
    /// Load an AMI model from a shared library path.
    ///
    /// # Safety
    /// The library must contain valid AMI_Init, AMI_GetWave, AMI_Close symbols.
    pub fn load(path: &Path) -> Result<Self, AmiLoadError> {
        // Validate file exists and has correct extension
        // Use libloading with RTLD_LOCAL to isolate symbol namespace
        let library = unsafe {
            libloading::Library::new(path)?
        };

        // Extract function pointers with validation
        let ami_init = unsafe { library.get(b"AMI_Init\0")? };
        let ami_getwave = unsafe { library.get(b"AMI_GetWave\0")? };
        let ami_close = unsafe { library.get(b"AMI_Close\0")? };

        Ok(Self { library, ami_init, ami_getwave, ami_close })
    }
}

// lifecycle.rs
pub struct AmiSession {
    library: Arc<AmiLibrary>,
    handle: AtomicPtr<c_void>,
    state: SessionState,
    resource_tracker: ResourceTracker,
}

#[derive(Clone, Copy, PartialEq)]
pub enum SessionState {
    Uninitialized,
    Initialized,
    Active,      // GetWave has been called at least once
    Closed,
    Faulted,     // Model crashed or timed out
}

impl AmiSession {
    pub fn init(
        &mut self,
        impulse: &mut Waveform,
        params: &AmiParameters,
        config: &SimulationConfig,
    ) -> Result<AmiInitResult, AmiError> {
        // Pre-call validation
        assert_eq!(self.state, SessionState::Uninitialized);

        // Prepare C-compatible buffers
        let mut impulse_buffer = impulse.samples.clone();
        let params_cstring = params.to_cstring()?;
        let mut params_out: *mut c_char = std::ptr::null_mut();
        let mut handle: *mut c_void = std::ptr::null_mut();
        let mut msg: *mut c_char = std::ptr::null_mut();

        // Execute with timeout and crash protection
        let result = self.execute_protected(|| unsafe {
            (self.library.ami_init)(
                impulse_buffer.as_mut_ptr(),
                impulse_buffer.len() as c_long,
                0, // aggressors
                impulse.dt.0,
                config.bit_time.0,
                params_cstring.as_ptr(),
                &mut params_out,
                &mut handle,
                &mut msg,
            )
        })?;

        // Store handle and transition state
        self.handle.store(handle, Ordering::SeqCst);
        self.state = SessionState::Initialized;

        // Copy modified impulse back
        impulse.samples.copy_from_slice(&impulse_buffer);

        // Track allocations for cleanup
        self.resource_tracker.track_output_string(params_out);
        self.resource_tracker.track_output_string(msg);

        Ok(AmiInitResult {
            return_code: result,
            output_params: Self::parse_output_params(params_out),
            message: Self::read_cstring(msg),
        })
    }

    pub fn getwave(
        &mut self,
        wave: &mut Waveform,
    ) -> Result<AmiGetWaveResult, AmiError> {
        assert!(matches!(self.state, SessionState::Initialized | SessionState::Active));

        let mut wave_buffer = wave.samples.clone();
        let mut clock_times = vec![0.0; wave.samples.len()];
        let mut params_out: *mut c_char = std::ptr::null_mut();

        let handle = self.handle.load(Ordering::SeqCst);

        let result = self.execute_protected(|| unsafe {
            (self.library.ami_getwave)(
                wave_buffer.as_mut_ptr(),
                wave_buffer.len() as c_long,
                clock_times.as_mut_ptr(),
                &mut params_out,
                handle,
            )
        })?;

        self.state = SessionState::Active;
        wave.samples.copy_from_slice(&wave_buffer);

        Ok(AmiGetWaveResult {
            return_code: result,
            clock_times,
            output_params: Self::parse_output_params(params_out),
        })
    }

    fn execute_protected<F, R>(&self, f: F) -> Result<R, AmiError>
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        // Option 1: Simple timeout with std::thread
        // Option 2: Process isolation (fork on Unix)
        // For Phase 1, we use a timeout-based approach

        let (tx, rx) = std::sync::mpsc::channel();
        let timeout = self.resource_tracker.timeout;

        std::thread::spawn(move || {
            // Install signal handlers for SIGSEGV, SIGFPE
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
            let _ = tx.send(result);
        });

        match rx.recv_timeout(timeout) {
            Ok(Ok(result)) => Ok(result),
            Ok(Err(_panic)) => Err(AmiError::ModelPanicked),
            Err(_) => Err(AmiError::Timeout),
        }
    }
}

impl Drop for AmiSession {
    fn drop(&mut self) {
        if matches!(self.state, SessionState::Initialized | SessionState::Active) {
            let handle = self.handle.load(Ordering::SeqCst);
            if !handle.is_null() {
                // Best-effort close, ignore errors
                let _ = std::panic::catch_unwind(|| unsafe {
                    (self.library.ami_close)(handle);
                });
            }
        }
        self.state = SessionState::Closed;
    }
}
```

### 5.4 Resource Tracking

```rust
// sandbox.rs
pub struct ResourceTracker {
    /// Maximum execution time for any single call
    pub timeout: Duration,
    /// Allocated strings that need to be freed
    output_strings: Mutex<Vec<*mut c_char>>,
    /// Peak memory usage tracking (if available)
    memory_limit: Option<usize>,
}

impl ResourceTracker {
    pub fn track_output_string(&self, ptr: *mut c_char) {
        if !ptr.is_null() {
            self.output_strings.lock().unwrap().push(ptr);
        }
    }

    /// Note: We cannot safely free vendor-allocated strings in the general case.
    /// This is tracked for debugging/auditing purposes.
    /// Vendors are responsible for cleanup in AMI_Close.
}
```

### 5.5 Platform-Specific Considerations

| Platform | Library Extension | Calling Convention | Notes |
|----------|------------------|-------------------|-------|
| Linux x86_64 | `.so` | System V AMD64 | Use `RTLD_LOCAL` |
| Windows x86_64 | `.dll` | Microsoft x64 | Use `LoadLibraryEx` with `LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR` |
| macOS ARM64 | `.dylib` | ARM64 | Rare in industry, but supported |

---

## 6. Convolution Engine

### 6.1 The Computational Challenge

For PCIe Gen 5 at 32 GT/s with BER target of 10^-12:
- Bit time: 31.25 ps
- Minimum bits for statistical confidence: ~10^13 bits
- At 32 GT/s: ~312 seconds of "real time" simulation

Direct time-domain convolution is computationally prohibitive. We employ two strategies:

1. **Statistical Mode**: Superposition of pulse responses (fast, approximate)
2. **Bit-by-Bit Mode**: Full simulation with parallel chunk processing

### 6.2 Statistical Convolution (Peak Distortion Analysis)

```rust
// convolution.rs

/// Statistical eye computation using superposition.
///
/// The pulse response is folded over itself at UI boundaries,
/// and all possible bit combinations are enumerated.
pub fn statistical_eye(
    pulse_response: &Waveform,
    config: &EyeConfig,
) -> StatisticalEye {
    let samples_per_ui = (config.bit_time / pulse_response.dt).0.round() as usize;
    let num_ui_in_pulse = pulse_response.samples.len() / samples_per_ui;

    // Fold pulse into UI-aligned cursors
    let cursors: Vec<Vec<f64>> = (0..samples_per_ui)
        .map(|phase| {
            (0..num_ui_in_pulse)
                .map(|ui| pulse_response.samples.get(phase + ui * samples_per_ui).copied().unwrap_or(0.0))
                .collect()
        })
        .collect();

    // Main cursor (largest magnitude, typically UI 0 or 1)
    let main_cursor_ui = find_main_cursor(&cursors);

    // Pre-cursor ISI (bits before main)
    // Post-cursor ISI (bits after main)
    // Peak distortion = sum of absolute ISI contributions

    let mut eye = StatisticalEye::new(samples_per_ui);

    // For each phase point in the UI
    for (phase, cursor_values) in cursors.iter().enumerate() {
        let main = cursor_values[main_cursor_ui];

        // ISI from pre-cursors
        let pre_isi: f64 = cursor_values[..main_cursor_ui].iter().map(|v| v.abs()).sum();

        // ISI from post-cursors
        let post_isi: f64 = cursor_values[main_cursor_ui + 1..].iter().map(|v| v.abs()).sum();

        // Worst case high/low levels
        eye.high[phase] = main + pre_isi + post_isi;
        eye.low[phase] = main - pre_isi - post_isi;
    }

    eye
}
```

### 6.3 Bit-by-Bit Convolution with Rayon

```rust
// convolution.rs

/// High-performance bit-by-bit convolution using overlap-save FFT method.
///
/// The bitstream is processed in chunks that can be parallelized across
/// CPU cores using Rayon.
pub struct ConvolutionEngine {
    /// Pre-computed FFT of impulse response (zero-padded)
    impulse_fft: Vec<Complex64>,
    /// FFT planner for forward/inverse transforms
    fft_forward: Arc<dyn Fft<f64>>,
    fft_inverse: Arc<dyn Fft<f64>>,
    /// Chunk size (power of 2, typically 2^16 to 2^20)
    chunk_size: usize,
    /// Overlap size (length of impulse - 1)
    overlap: usize,
}

impl ConvolutionEngine {
    pub fn new(impulse: &Waveform) -> Self {
        // Choose chunk size: balance between FFT efficiency and memory
        let impulse_len = impulse.samples.len();
        let chunk_size = (impulse_len * 4).next_power_of_two().max(65536);
        let overlap = impulse_len - 1;

        // Pre-compute impulse FFT
        let mut planner = FftPlanner::new();
        let fft_forward = planner.plan_fft_forward(chunk_size);
        let fft_inverse = planner.plan_fft_inverse(chunk_size);

        let mut impulse_padded: Vec<Complex64> = impulse.samples
            .iter()
            .map(|&v| Complex64 { re: v, im: 0.0 })
            .collect();
        impulse_padded.resize(chunk_size, Complex64 { re: 0.0, im: 0.0 });

        fft_forward.process(&mut impulse_padded);

        Self {
            impulse_fft: impulse_padded,
            fft_forward,
            fft_inverse,
            chunk_size,
            overlap,
        }
    }

    /// Convolve an entire bitstream, returning the output waveform.
    /// Uses Rayon for parallel chunk processing.
    pub fn convolve(&self, input: &[f64]) -> Vec<f64> {
        let valid_chunk_size = self.chunk_size - self.overlap;
        let num_chunks = (input.len() + valid_chunk_size - 1) / valid_chunk_size;

        // Process chunks in parallel
        let chunk_results: Vec<Vec<f64>> = (0..num_chunks)
            .into_par_iter()
            .map(|chunk_idx| {
                let start = chunk_idx * valid_chunk_size;
                let end = (start + self.chunk_size).min(input.len());

                self.convolve_chunk(&input[start..end])
            })
            .collect();

        // Stitch results with overlap-add
        self.overlap_add(chunk_results, input.len())
    }

    fn convolve_chunk(&self, chunk: &[f64]) -> Vec<f64> {
        // Zero-pad input chunk
        let mut chunk_fft: Vec<Complex64> = chunk
            .iter()
            .map(|&v| Complex64 { re: v, im: 0.0 })
            .collect();
        chunk_fft.resize(self.chunk_size, Complex64 { re: 0.0, im: 0.0 });

        // Forward FFT
        self.fft_forward.process(&mut chunk_fft);

        // Multiply with impulse FFT (frequency domain convolution)
        for (c, h) in chunk_fft.iter_mut().zip(self.impulse_fft.iter()) {
            *c = *c * *h;
        }

        // Inverse FFT
        self.fft_inverse.process(&mut chunk_fft);

        // Extract real part, normalize
        let scale = 1.0 / self.chunk_size as f64;
        chunk_fft.iter().map(|c| c.re * scale).collect()
    }

    fn overlap_add(&self, chunks: Vec<Vec<f64>>, total_len: usize) -> Vec<f64> {
        let valid_chunk_size = self.chunk_size - self.overlap;
        let output_len = total_len + self.overlap;
        let mut output = vec![0.0; output_len];

        for (idx, chunk) in chunks.into_iter().enumerate() {
            let start = idx * valid_chunk_size;
            for (i, &val) in chunk.iter().enumerate() {
                if start + i < output.len() {
                    output[start + i] += val;
                }
            }
        }

        output
    }
}
```

### 6.4 GPU Acceleration (Phase 2 - Deferred)

The WebGPU/wgpu approach is deferred due to:

1. **Memory Transfer Latency**: AMI models run on CPU. Each GetWave call would require:
   - CPU → GPU transfer of waveform
   - GPU convolution
   - GPU → CPU transfer for AMI processing

2. **Synchronization Overhead**: Back-channel adaptation requires Tx/Rx state synchronization, forcing GPU→CPU roundtrips per adaptation cycle.

3. **Recommendation**: GPU acceleration is viable for:
   - Post-AMI pure channel convolution (no equalization state)
   - Eye diagram rendering
   - Monte Carlo BER estimation

---

## 7. S-Parameter Processing

### 7.1 Touchstone Parsing

```rust
// touchstone.rs in lib-ibis

pub struct TouchstoneFile {
    pub version: TouchstoneVersion,
    pub num_ports: usize,
    pub format: DataFormat,      // RI, MA, DB
    pub frequency_unit: Hertz,   // Multiplier (Hz, kHz, MHz, GHz)
    pub z0: Ohms,
    pub data: Vec<FrequencyPoint>,
}

pub struct FrequencyPoint {
    pub frequency: Hertz,
    pub s_matrix: SMatrix,
}

pub fn parse_touchstone(content: &str) -> Result<TouchstoneFile, ParseError> {
    // nom-based parser for .s2p, .s4p, .s8p, etc.
    // Handle all standard formats: RI (real/imag), MA (mag/angle), DB (dB/angle)
}
```

### 7.2 S-Parameter to Pulse Response Conversion

```rust
// fft.rs in lib-dsp

/// Convert S-parameters (frequency domain) to pulse response (time domain).
///
/// For a differential channel, we typically use SDD21 (differential through).
pub fn sparam_to_pulse(
    sparams: &SParameters,
    config: &ConversionConfig,
) -> Result<Waveform, DspError> {
    // 1. Select the relevant S-parameter (e.g., S21 or SDD21)
    let transfer_function: Vec<Complex64> = sparams.matrices
        .iter()
        .map(|m| m[[config.output_port, config.input_port]])
        .collect();

    // 2. Interpolate to uniform frequency grid
    let uniform_freqs = generate_uniform_frequencies(
        sparams.frequencies.first().unwrap().0,
        sparams.frequencies.last().unwrap().0,
        config.num_fft_points,
    );
    let interpolated = interpolate_sparam(&sparams.frequencies, &transfer_function, &uniform_freqs)?;

    // 3. Enforce causality (see section 7.3)
    let causal = enforce_causality(&interpolated)?;

    // 4. Enforce passivity (see section 7.4)
    let passive = enforce_passivity(&causal)?;

    // 5. IFFT to time domain
    let mut fft_buffer: Vec<Complex64> = passive;

    // Hermitian symmetry for real output
    let n = fft_buffer.len();
    for i in 1..n/2 {
        fft_buffer[n - i] = fft_buffer[i].conj();
    }

    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(n);
    ifft.process(&mut fft_buffer);

    // 6. Extract real part as impulse response
    let dt = Seconds(1.0 / (2.0 * uniform_freqs.last().unwrap().0));
    let samples: Vec<f64> = fft_buffer.iter().map(|c| c.re / n as f64).collect();

    // 7. Integrate to get pulse response (step response convolved with rect pulse)
    let pulse = impulse_to_pulse(&samples, config.bit_time, dt);

    Ok(Waveform {
        samples: pulse,
        dt,
        t_start: Seconds(0.0),
    })
}
```

### 7.3 Causality Enforcement

A causal system has zero response before t=0. In frequency domain, this requires the real and imaginary parts to satisfy the Hilbert transform relationship.

```rust
// causality.rs

/// Enforce causality using the Hilbert transform relationship.
///
/// For a causal system: Im(H) = -Hilbert(Re(H))
/// This ensures the impulse response is zero for t < 0.
pub fn enforce_causality(h: &[Complex64]) -> Result<Vec<Complex64>, DspError> {
    let n = h.len();

    // Extract magnitude (we preserve magnitude, adjust phase)
    let magnitudes: Vec<f64> = h.iter().map(|c| c.norm()).collect();

    // Compute minimum-phase response via cepstrum
    // ln(|H|) -> IFFT -> window -> FFT -> exp
    let log_mag: Vec<f64> = magnitudes.iter().map(|m| (m + 1e-15).ln()).collect();

    // IFFT of log magnitude
    let mut cepstrum: Vec<Complex64> = log_mag
        .iter()
        .map(|&v| Complex64 { re: v, im: 0.0 })
        .collect();

    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(n);
    ifft.process(&mut cepstrum);

    // Causal windowing: keep n=0, double n=1..N/2-1, zero n=N/2..N-1
    cepstrum[0].re /= n as f64;
    cepstrum[0].im = 0.0;
    for i in 1..n/2 {
        cepstrum[i].re *= 2.0 / n as f64;
        cepstrum[i].im *= 2.0 / n as f64;
    }
    for i in n/2..n {
        cepstrum[i] = Complex64 { re: 0.0, im: 0.0 };
    }

    // FFT back
    let fft = planner.plan_fft_forward(n);
    fft.process(&mut cepstrum);

    // Exponentiate to get minimum-phase response
    let causal: Vec<Complex64> = cepstrum
        .iter()
        .map(|c| {
            let mag = c.re.exp();
            let phase = c.im;
            Complex64 {
                re: mag * phase.cos(),
                im: mag * phase.sin(),
            }
        })
        .collect();

    Ok(causal)
}
```

### 7.4 Passivity Enforcement

A passive network cannot generate energy. This requires |S21| <= 1 at all frequencies.

```rust
// passivity.rs

/// Enforce passivity on S-parameters.
///
/// For a passive network: ||S|| <= 1 (all singular values <= 1)
/// For simple 2-port: |S11|^2 + |S21|^2 <= 1 and |S22|^2 + |S12|^2 <= 1
pub fn enforce_passivity(sparams: &mut SParameters) -> Result<(), DspError> {
    for matrix in &mut sparams.matrices {
        enforce_passivity_matrix(matrix)?;
    }
    Ok(())
}

fn enforce_passivity_matrix(s: &mut SMatrix) -> Result<(), DspError> {
    let n = s.nrows();

    // Compute S^H * S (should have eigenvalues <= 1 for passive network)
    let s_h = s.t().mapv(|c| c.conj());
    let product = s_h.dot(s);

    // Compute eigenvalues (for 2x2, use direct formula; larger uses LAPACK)
    let eigenvalues = compute_eigenvalues(&product)?;

    let max_eigenvalue = eigenvalues.iter().map(|e| e.re).fold(0.0, f64::max);

    if max_eigenvalue > 1.0 {
        // Scale S-matrix to enforce passivity
        let scale = 1.0 / max_eigenvalue.sqrt();
        s.mapv_inplace(|c| Complex64 {
            re: c.re * scale,
            im: c.im * scale,
        });

        tracing::warn!(
            "S-parameter passivity violated (max eigenvalue = {:.4}), scaled by {:.4}",
            max_eigenvalue,
            scale
        );
    }

    Ok(())
}

/// Alternative: Iterative passivity enforcement preserving frequency response shape.
/// Uses perturbation to minimally modify the S-parameters.
pub fn enforce_passivity_iterative(
    sparams: &mut SParameters,
    max_iterations: usize,
    tolerance: f64,
) -> Result<PassivityReport, DspError> {
    // Implementation of Hamiltonian matrix perturbation method
    // Reference: Gustavsen & Semlyen, "Enforcing Passivity for Admittance Matrices"
    todo!("Implement Hamiltonian perturbation method")
}
```

---

## 8. Back-Channel Training Architecture

### 8.1 PCIe Link Training Overview

PCIe Gen 5 uses a sophisticated link training sequence where Tx and Rx negotiate:
- **Tx Preset (Pn)**: Pre-defined Tx equalization settings (P0-P10)
- **Rx Hint**: Rx feedback on which preset works best
- **Adaptation**: Fine-tuning of Tx/Rx EQ coefficients

### 8.2 Message Bus Design

```rust
// backchannel.rs in lib-ami-ffi

/// Message types for Tx-Rx back-channel communication.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum BackChannelMessage {
    /// Rx requests Tx to change preset
    PresetRequest { preset: u8 },

    /// Rx reports current figure of merit
    FigureOfMerit { value: f64 },

    /// Tx reports current coefficient values
    TxCoefficients {
        pre_cursor: f64,
        main_cursor: f64,
        post_cursor: f64,
    },

    /// Rx reports eye margin measurements
    EyeMargin {
        height_mv: f64,
        width_ps: f64,
    },

    /// Generic parameter update (model-specific)
    ParameterUpdate {
        name: String,
        value: AmiValue,
    },

    /// Training complete signal
    TrainingComplete,
}

/// The message bus that coordinates Tx and Rx communication.
pub struct BackChannelBus {
    /// Messages from Rx to Tx
    rx_to_tx: Mutex<VecDeque<BackChannelMessage>>,
    /// Messages from Tx to Rx
    tx_to_rx: Mutex<VecDeque<BackChannelMessage>>,
    /// Current training state
    state: AtomicU8,
}

#[repr(u8)]
pub enum TrainingState {
    Idle = 0,
    PresetSweep = 1,
    CoarseAdaptation = 2,
    FineAdaptation = 3,
    Converged = 4,
}

impl BackChannelBus {
    pub fn new() -> Self {
        Self {
            rx_to_tx: Mutex::new(VecDeque::new()),
            tx_to_rx: Mutex::new(VecDeque::new()),
            state: AtomicU8::new(TrainingState::Idle as u8),
        }
    }

    /// Called by Rx AMI model to send message to Tx
    pub fn rx_send(&self, msg: BackChannelMessage) {
        self.rx_to_tx.lock().unwrap().push_back(msg);
    }

    /// Called by Tx AMI model to receive messages from Rx
    pub fn tx_receive(&self) -> Option<BackChannelMessage> {
        self.rx_to_tx.lock().unwrap().pop_front()
    }

    /// Called by Tx AMI model to send message to Rx
    pub fn tx_send(&self, msg: BackChannelMessage) {
        self.tx_to_rx.lock().unwrap().push_back(msg);
    }

    /// Called by Rx AMI model to receive messages from Tx
    pub fn rx_receive(&self) -> Option<BackChannelMessage> {
        self.tx_to_rx.lock().unwrap().pop_front()
    }
}
```

### 8.3 Training State Machine

```rust
// orchestrator.rs in kernel-cli

pub struct TrainingOrchestrator {
    tx_session: AmiSession,
    rx_session: AmiSession,
    bus: Arc<BackChannelBus>,
    config: TrainingConfig,
}

impl TrainingOrchestrator {
    /// Execute the link training sequence.
    pub fn train(&mut self, channel: &Waveform) -> Result<TrainingResult, SimError> {
        let mut best_preset = 0u8;
        let mut best_fom = f64::NEG_INFINITY;

        // Phase 1: Preset Sweep
        self.bus.set_state(TrainingState::PresetSweep);

        for preset in 0..=10 {
            // Configure Tx with this preset
            self.tx_session.set_parameter("preset", AmiValue::Integer(preset as i64))?;

            // Run short simulation
            let eye = self.simulate_eye(channel, 1_000_000)?; // 1M bits

            // Measure figure of merit
            let fom = eye.figure_of_merit();

            if fom > best_fom {
                best_fom = fom;
                best_preset = preset;
            }
        }

        // Configure best preset
        self.tx_session.set_parameter("preset", AmiValue::Integer(best_preset as i64))?;

        // Phase 2: Fine Adaptation (if supported by models)
        if self.supports_adaptation() {
            self.bus.set_state(TrainingState::FineAdaptation);

            for iteration in 0..self.config.max_adaptation_iterations {
                // Run simulation chunk
                let mut wave = self.generate_prbs(self.config.adaptation_chunk_bits);

                // Tx processes wave
                self.tx_session.getwave(&mut wave)?;

                // Channel convolution
                let convolved = self.convolve(channel, &wave);

                // Rx processes and adapts
                let mut rx_wave = convolved.clone();
                let rx_result = self.rx_session.getwave(&mut rx_wave)?;

                // Extract Rx feedback from output parameters
                if let Some(feedback) = self.extract_rx_feedback(&rx_result) {
                    self.bus.rx_send(feedback);
                }

                // Deliver feedback to Tx for next iteration
                while let Some(msg) = self.bus.tx_receive() {
                    self.apply_tx_feedback(&msg)?;
                }

                // Check convergence
                if self.is_converged() {
                    break;
                }
            }
        }

        self.bus.set_state(TrainingState::Converged);

        Ok(TrainingResult {
            final_preset: best_preset,
            final_fom: best_fom,
            tx_coefficients: self.tx_session.get_coefficients(),
            rx_coefficients: self.rx_session.get_coefficients(),
        })
    }
}
```

### 8.4 AMI InOut Parameter Mapping

The IBIS-AMI spec defines `InOut` parameters that can be modified during simulation:

```
(Reserved_Parameters
    (Tx_Tap_Units UI)
    (Tx_Tap (-0.1 -0.05 0.0) (Range -0.2 0.2) (Description "Tx pre-cursor tap"))
)
```

Our kernel maps these to the back-channel:

```rust
// Extract InOut parameters from AMI output string
fn parse_ami_output_params(output: &str) -> HashMap<String, AmiValue> {
    // Parse the Lisp-like syntax from AMI_GetWave output
    // Map to internal parameter representation
}

// Inject updated parameters into AMI input string
fn inject_ami_input_params(
    base_params: &str,
    updates: &HashMap<String, AmiValue>,
) -> String {
    // Modify the parameter tree with new values
}
```

---

## 9. Simulation Modes

### 9.1 Mode Selection

```rust
// config.rs

#[derive(Clone, Debug)]
pub enum SimulationMode {
    /// Statistical mode: fast, uses superposition
    Statistical {
        /// Number of phase points to compute
        phase_points: usize,
    },

    /// Bit-by-bit: full time-domain simulation
    BitByBit {
        /// Number of bits to simulate
        num_bits: u64,
        /// PRBS pattern order
        prbs_order: u8,
    },

    /// Hybrid: statistical for initial analysis, bit-by-bit for verification
    Hybrid {
        statistical_config: Box<SimulationMode>,
        verification_bits: u64,
    },
}
```

### 9.2 PRBS Pattern Generation

```rust
// prbs.rs

pub struct PrbsGenerator {
    state: u64,
    taps: u64,
    length: usize,
}

impl PrbsGenerator {
    /// Create a PRBS-N generator.
    /// Supported orders: 7, 9, 11, 15, 23, 31
    pub fn new(order: u8) -> Self {
        let (taps, length) = match order {
            7 => (0b1100000, (1 << 7) - 1),
            9 => (0b100010000, (1 << 9) - 1),
            11 => (0b10100000000, (1 << 11) - 1),
            15 => (0b110000000000000, (1 << 15) - 1),
            23 => (0b100001000000000000000, (1 << 23) - 1),
            31 => (0b1001000000000000000000000000000, (1 << 31) - 1),
            _ => panic!("Unsupported PRBS order: {}", order),
        };

        Self {
            state: 1, // Non-zero initial state
            taps,
            length,
        }
    }

    pub fn next_bit(&mut self) -> u8 {
        let feedback = (self.state & self.taps).count_ones() & 1;
        self.state = (self.state >> 1) | ((feedback as u64) << (self.length.trailing_zeros() as u64));
        (self.state & 1) as u8
    }

    /// Generate a voltage waveform from bits
    pub fn generate_waveform(&mut self, num_bits: u64, config: &WaveformConfig) -> Waveform {
        let samples_per_bit = (config.sample_rate * config.bit_time.0) as usize;
        let total_samples = num_bits as usize * samples_per_bit;

        let mut samples = Vec::with_capacity(total_samples);

        for _ in 0..num_bits {
            let bit = self.next_bit();
            let voltage = if bit == 1 { config.v_high } else { config.v_low };

            // Apply rise/fall time shaping if configured
            for _ in 0..samples_per_bit {
                samples.push(voltage);
            }
        }

        Waveform {
            samples,
            dt: Seconds(1.0 / config.sample_rate),
            t_start: Seconds(0.0),
        }
    }
}
```

---

## 10. Error Handling Strategy

### 10.1 Error Types Hierarchy

```rust
// In each crate's lib.rs

// lib-types: No errors (pure data)

// lib-ibis
#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    #[error("Invalid IBIS syntax at line {line}: {message}")]
    SyntaxError { line: usize, message: String },

    #[error("Missing required section: {section}")]
    MissingSection { section: String },

    #[error("Invalid S-parameter format: {0}")]
    InvalidSParamFormat(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

// lib-ami-ffi
#[derive(Debug, thiserror::Error)]
pub enum AmiError {
    #[error("Failed to load library: {0}")]
    LoadError(#[from] libloading::Error),

    #[error("AMI_Init returned error code {code}: {message}")]
    InitFailed { code: i64, message: String },

    #[error("AMI_GetWave returned error code {code}")]
    GetWaveFailed { code: i64 },

    #[error("Model execution timed out after {0:?}")]
    Timeout(Duration),

    #[error("Model crashed: {0}")]
    ModelPanicked(String),

    #[error("Invalid model state: expected {expected:?}, got {actual:?}")]
    InvalidState { expected: SessionState, actual: SessionState },
}

// lib-dsp
#[derive(Debug, thiserror::Error)]
pub enum DspError {
    #[error("FFT size must be power of 2, got {0}")]
    InvalidFftSize(usize),

    #[error("Causality enforcement failed: {0}")]
    CausalityEnforcementFailed(String),

    #[error("Passivity enforcement failed: {0}")]
    PassivityEnforcementFailed(String),

    #[error("Interpolation failed: insufficient frequency points")]
    InterpolationFailed,
}

// kernel-cli
#[derive(Debug, thiserror::Error)]
pub enum SimError {
    #[error("Parse error: {0}")]
    Parse(#[from] ParseError),

    #[error("AMI error: {0}")]
    Ami(#[from] AmiError),

    #[error("DSP error: {0}")]
    Dsp(#[from] DspError),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Training failed to converge after {iterations} iterations")]
    TrainingNotConverged { iterations: usize },
}
```

---

## 11. Performance Considerations

### 11.1 Memory Layout

- All waveforms use contiguous `Vec<f64>` for cache efficiency
- S-parameter matrices use `ndarray` with column-major layout (BLAS-compatible)
- FFT buffers are pre-allocated and reused via `Arc<Mutex<...>>` pools

### 11.2 Parallelization Strategy

| Operation | Parallelization | Rationale |
|-----------|----------------|-----------|
| File parsing | Single-threaded | I/O bound |
| S-param interpolation | Rayon (per frequency) | CPU bound |
| Passivity enforcement | Single-threaded | Matrix ops already vectorized |
| Convolution chunks | Rayon | Embarrassingly parallel |
| Eye diagram binning | Rayon | Embarrassingly parallel |
| AMI GetWave | Sequential | Vendor binary is stateful |

### 11.3 Benchmarking Targets

| Operation | Target | Notes |
|-----------|--------|-------|
| Parse 10MB .s4p | < 100ms | nom is fast |
| 1M point FFT | < 10ms | rustfft is competitive with FFTW |
| 10^9 bit convolution | < 60s | With 16 cores |
| Eye diagram (1M UIs) | < 5s | Binning is O(n) |

---

## 12. Security Model

### 12.1 Threat Model

Vendor AMI binaries are untrusted code. Potential threats:
- Arbitrary code execution
- Information disclosure (reading other files)
- Resource exhaustion (CPU, memory)
- System modification

### 12.2 Mitigations

| Threat | Mitigation | Implementation |
|--------|------------|----------------|
| Code execution | Sandboxing | Phase 2: seccomp/AppArmor |
| Memory exhaustion | Resource limits | `setrlimit` on Linux |
| CPU exhaustion | Timeouts | `recv_timeout` on thread |
| File access | Chroot/namespace | Phase 2: Linux namespaces |
| Network access | Firewall | Drop capabilities |

### 12.3 Phase 1 Baseline

For initial release, we implement:
- Execution timeouts
- Panic catching (`catch_unwind`)
- Memory tracking (advisory)

Phase 2 will add process isolation via `fork()` or dedicated sandbox processes.

---

## 13. Future Extensions

### 13.1 Near-term (v0.2)

- [ ] PAM4 support for PCIe Gen 6
- [ ] Crosstalk analysis (FEXT/NEXT)
- [ ] CTLE/DFE modeling in pure Rust (for channel-only analysis)
- [ ] Waveform visualization (ratatui TUI or SVG export)

### 13.2 Medium-term (v0.3)

- [ ] GPU acceleration for large-scale Monte Carlo
- [ ] Python bindings (PyO3)
- [ ] Process-isolated sandbox for AMI binaries
- [ ] Distributed simulation (multiple channels in parallel)

### 13.3 Long-term (v1.0)

- [ ] DDR5/LPDDR5 support
- [ ] USB4/Thunderbolt support
- [ ] Custom IBIS model creation tools
- [ ] Cloud-native deployment (containerized workers)

---

## Appendix A: References

1. IBIS 7.2 Specification (ibis.org)
2. IBIS-AMI Modeling Cookbook (sisoft.com)
3. "High-Speed Digital Design" - Johnson & Graham
4. "Signal and Power Integrity - Simplified" - Bogatin
5. PCIe Base Specification 5.0/6.0 (pcisig.com)

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| AMI | Algorithmic Modeling Interface |
| BER | Bit Error Rate |
| CDR | Clock and Data Recovery |
| CTLE | Continuous Time Linear Equalizer |
| DFE | Decision Feedback Equalizer |
| FFE | Feed-Forward Equalizer |
| IBIS | I/O Buffer Information Specification |
| ISI | Inter-Symbol Interference |
| NRZ | Non-Return-to-Zero (2-level signaling) |
| PAM4 | Pulse Amplitude Modulation 4-level |
| PRBS | Pseudo-Random Bit Sequence |
| UI | Unit Interval (1 bit period) |

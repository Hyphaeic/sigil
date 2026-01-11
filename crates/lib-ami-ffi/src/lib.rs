//! # lib-ami-ffi
//!
//! Safe FFI wrappers for IBIS-AMI vendor binaries.
//!
//! This crate provides a safe Rust interface for loading and executing
//! vendor-supplied AMI models (`.dll`/`.so` files). It handles:
//!
//! - Dynamic library loading with `libloading`
//! - AMI lifecycle management (Init/GetWave/Close)
//! - Timeout protection and crash recovery
//! - Resource tracking and cleanup
//! - Back-channel communication for Tx-Rx adaptation
//!
//! # Safety
//!
//! Vendor binaries are untrusted code. This crate implements multiple
//! safety layers:
//!
//! 1. **Timeout protection**: All calls are wrapped with configurable timeouts
//! 2. **Panic catching**: `catch_unwind` prevents panics from unwinding into caller
//! 3. **Resource tracking**: All allocations are tracked for cleanup
//! 4. **State machine**: Session states prevent invalid call sequences
//!
//! Future versions may add process isolation via fork() or sandboxing.

pub mod error;
pub mod loader;
pub mod lifecycle;
pub mod backchannel;

pub use error::AmiError;
pub use loader::AmiLibrary;
pub use lifecycle::{AmiSession, SessionState};
pub use backchannel::{BackChannelBus, TrainingState};

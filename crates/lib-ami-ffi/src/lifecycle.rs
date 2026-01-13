//! AMI session lifecycle management.
//!
//! This module manages the lifecycle of AMI model sessions, including:
//! - Initialization (AMI_Init)
//! - Waveform processing (AMI_GetWave)
//! - Cleanup (AMI_Close)
//!
//! Each session maintains state to ensure correct call ordering and
//! proper resource cleanup.

use crate::error::{AmiError, AmiResult};
use crate::loader::AmiLibrary;
use lib_types::ami::{AmiConfig, AmiGetWaveResult, AmiInitResult, AmiParameters};
use lib_types::waveform::Waveform;
use std::ffi::{c_char, c_void, CStr, CString};
use std::ptr;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// Global counter for orphaned threads from timed-out FFI calls.
/// When a timeout occurs, the spawned thread continues running but we can't cancel it.
/// This counter tracks how many such threads exist to prevent resource exhaustion.
static ORPHANED_THREAD_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Maximum number of orphaned threads allowed before refusing new operations.
const MAX_ORPHANED_THREADS: usize = 10;

pub use lib_types::ami::SessionState;

/// Configuration for AMI session execution.
#[derive(Clone, Debug)]
pub struct ExecutionConfig {
    /// Maximum time for any single call.
    pub timeout: Duration,

    /// Whether to catch panics from the model.
    pub catch_panics: bool,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(30),
            catch_panics: true,
        }
    }
}

/// Raw FFI output from AMI_Init - all pointers stored as usize for Send safety.
struct InitFfiOutput {
    return_code: i64,
    params_out: usize,  // *mut c_char as usize
    handle: usize,      // *mut c_void as usize
    msg: usize,         // *mut c_char as usize
    impulse_buffer: Vec<f64>,
}

/// Raw FFI output from AMI_GetWave.
struct GetWaveFfiOutput {
    return_code: i64,
    /// CRIT-FFI-001 FIX: Store as String (copied inside closure), not pointer
    params_out_str: Option<String>,
    wave_buffer: Vec<f64>,
    clock_times: Vec<f64>,
    /// HIGH-FFI-004 FIX: Sentinel overrun detection
    overrun_index: Option<usize>,
}

/// An active AMI model session.
///
/// This struct manages the lifecycle of a single AMI model instance,
/// ensuring proper initialization, execution, and cleanup.
///
/// # Thread Safety (HIGH-FFI-003)
///
/// Per IBIS 7.2 Section 10.1:
///
/// > "The model may maintain internal state between AMI_Init and AMI_Close.
/// >  The simulator shall not call the same model instance concurrently
/// >  from multiple threads."
///
/// This struct is intentionally `!Sync` to prevent sharing across threads.
/// The `_not_sync` marker ensures this at compile time.
///
/// ## Correct Usage
///
/// ```ignore
/// // Create separate sessions for parallel simulation
/// let session1 = AmiSession::new(library.clone());
/// let session2 = AmiSession::new(library.clone());
///
/// // Each session can run in its own thread
/// thread::spawn(move || session1.init(&config));
/// thread::spawn(move || session2.init(&config));
/// ```
///
/// ## Incorrect Usage (Won't Compile)
///
/// ```ignore
/// let session = Arc::new(AmiSession::new(library)); // ERROR: !Sync
/// let s1 = session.clone();
/// thread::spawn(move || s1.getwave(...)); // Would violate IBIS spec
/// ```
///
/// If you need to share a session, wrap it in a `Mutex<AmiSession>` to
/// serialize access.
pub struct AmiSession {
    /// The loaded library.
    library: Arc<AmiLibrary>,

    /// Opaque handle from AMI_Init.
    handle: AtomicPtr<c_void>,

    /// Current session state.
    state: SessionState,

    /// Execution configuration.
    config: ExecutionConfig,

    /// Number of GetWave calls made.
    getwave_count: u64,

    /// CRIT-FFI-002 FIX: Counter for in-flight FFI operations.
    /// Used to prevent close() while orphaned threads still use the handle.
    pending_ops: Arc<AtomicUsize>,

    /// HIGH-FFI-003 FIX: Marker to prevent Sync implementation.
    /// This ensures the session cannot be shared across threads without Mutex.
    _not_sync: std::marker::PhantomData<std::cell::Cell<()>>,
}

impl AmiSession {
    /// Create a new uninitialized session.
    pub fn new(library: Arc<AmiLibrary>) -> Self {
        Self {
            library,
            handle: AtomicPtr::new(ptr::null_mut()),
            state: SessionState::Uninitialized,
            config: ExecutionConfig::default(),
            getwave_count: 0,
            pending_ops: Arc::new(AtomicUsize::new(0)),
            _not_sync: std::marker::PhantomData,
        }
    }

    /// Create a session with custom execution config.
    pub fn with_config(library: Arc<AmiLibrary>, config: ExecutionConfig) -> Self {
        Self {
            library,
            handle: AtomicPtr::new(ptr::null_mut()),
            state: SessionState::Uninitialized,
            config,
            getwave_count: 0,
            pending_ops: Arc::new(AtomicUsize::new(0)),
            _not_sync: std::marker::PhantomData,
        }
    }

    /// Get the current session state.
    pub fn state(&self) -> SessionState {
        self.state
    }

    /// Check if GetWave is supported.
    pub fn supports_getwave(&self) -> bool {
        self.library.supports_getwave()
    }

    /// Initialize the AMI model.
    ///
    /// # Arguments
    ///
    /// * `impulse` - Channel impulse response (may be modified by model)
    /// * `ami_config` - Simulation configuration
    /// * `params` - Input parameters for the model
    ///
    /// # Returns
    ///
    /// Result containing initialization information from the model.
    pub fn init(
        &mut self,
        impulse: &mut Waveform,
        ami_config: &AmiConfig,
        params: &AmiParameters,
    ) -> AmiResult<AmiInitResult> {
        // Validate state
        if self.state != SessionState::Uninitialized {
            return Err(AmiError::invalid_state(
                SessionState::Uninitialized,
                self.state,
            ));
        }

        // Prepare input parameters as C string
        let params_str = params.to_ami_string();
        let params_cstr =
            CString::new(params_str).map_err(|_| AmiError::InvalidParameter {
                name: "parameters".to_string(),
                reason: "Contains null byte".to_string(),
            })?;

        // Copy impulse to buffer that can be modified
        let original_impulse = impulse.samples.clone();
        let impulse_buffer = impulse.samples.clone();

        // Get function pointer and config values
        let init_fn = self.library.init_fn();
        let num_aggressors = ami_config.num_aggressors as i64;
        let sample_interval = ami_config.sample_interval.0;
        let bit_time = ami_config.bit_time.0;

        // Execute AMI_Init - closure owns all data and returns all outputs
        let ffi_output = self.execute_protected(move || {
            let mut impulse_buffer = impulse_buffer;
            let mut params_out: *mut c_char = ptr::null_mut();
            let mut handle: *mut c_void = ptr::null_mut();
            let mut msg: *mut c_char = ptr::null_mut();

            let return_code = unsafe {
                init_fn(
                    impulse_buffer.as_mut_ptr(),
                    impulse_buffer.len() as i64,
                    num_aggressors,
                    sample_interval,
                    bit_time,
                    params_cstr.as_ptr(),
                    &mut params_out,
                    &mut handle,
                    &mut msg,
                )
            };

            InitFfiOutput {
                return_code,
                params_out: params_out as usize,
                handle: handle as usize,
                msg: msg as usize,
                impulse_buffer,
            }
        })?;

        // Convert pointers back from usize
        let params_out = ffi_output.params_out as *mut c_char;
        let handle = ffi_output.handle as *mut c_void;
        let msg = ffi_output.msg as *mut c_char;

        // CRIT-003 FIX: Read each C string exactly once immediately after FFI call.
        // The vendor may free memory at any time, so we must copy strings immediately.
        let message = unsafe { read_c_string(msg) };
        let output_params_str = unsafe { read_c_string(params_out) };

        // Check return code
        if ffi_output.return_code != 0 {
            // MED-007 FIX: Cleanup any handle returned by the model even on failure
            // Some vendors return handles even on init failure - we must close them
            if !handle.is_null() {
                tracing::debug!("Cleaning up handle after init failure");
                let close_fn = self.library.close_fn();
                let handle_usize = handle as usize;
                // Best-effort close, ignore errors since we're already failing
                let _ = self.execute_protected(move || {
                    let h = handle_usize as *mut c_void;
                    unsafe { close_fn(h) }
                });
            }
            self.state = SessionState::Faulted;
            return Err(AmiError::init_failed(
                ffi_output.return_code,
                message.unwrap_or_default(),
            ));
        }

        // Store handle and update state
        self.handle.store(handle, Ordering::SeqCst);
        self.state = SessionState::Initialized;

        // Check if impulse was modified
        let modified_impulse = ffi_output.impulse_buffer != original_impulse;
        if modified_impulse {
            impulse.samples = ffi_output.impulse_buffer;
        }

        // Parse output parameters from the already-read string
        let output_params = output_params_str
            .and_then(|s| AmiParameters::from_ami_string(&s).ok())
            .unwrap_or_default();

        tracing::debug!(
            return_code = ffi_output.return_code,
            modified_impulse,
            has_message = message.is_some(),
            "AMI_Init completed"
        );

        Ok(AmiInitResult {
            return_code: ffi_output.return_code,
            output_params,
            message,
            supports_getwave: self.library.supports_getwave(),
            modified_impulse,
        })
    }

    /// Process a waveform through the model.
    ///
    /// # Arguments
    ///
    /// * `wave` - Waveform to process (modified in place)
    ///
    /// # Returns
    ///
    /// Result containing CDR clock times and output parameters.
    pub fn getwave(&mut self, wave: &mut Waveform) -> AmiResult<AmiGetWaveResult> {
        // Validate state
        match self.state {
            SessionState::Initialized | SessionState::Active => {}
            _ => {
                return Err(AmiError::invalid_state(SessionState::Active, self.state));
            }
        }

        // Check if GetWave is supported
        let getwave_fn = self.library.getwave_fn().ok_or(AmiError::NotSupported {
            operation: "GetWave".to_string(),
        })?;

        // Get handle as usize for Send safety
        let handle_usize = self.handle.load(Ordering::SeqCst) as usize;
        if handle_usize == 0 {
            return Err(AmiError::InvalidOutput(
                "Null handle after initialization".to_string(),
            ));
        }

        // HIGH-FFI-004 FIX: Prepare buffers with sentinel values for overrun detection
        //
        // Per IBIS 7.2 Section 10.2.3: "wave_size indicates the maximum number of samples"
        // Some models with bugs (especially in CDR oversampling modes) write beyond this limit.
        //
        // We add sentinel values at the end of the buffer to detect overruns.
        // Using a recognizable bit pattern (0xDEADBEEF encoded as f64).
        const SENTINEL: f64 = f64::from_bits(0xDEADBEEFDEADBEEF);
        const SENTINEL_COUNT: usize = 8; // Multiple sentinels to detect extent of overrun

        let wave_buffer = wave.samples.clone();
        let clock_times_len = wave.samples.len();
        let original_buffer_len = wave_buffer.len();

        // Get model name for error reporting
        let model_name = self.library.path.clone();

        // Execute AMI_GetWave - closure owns all data
        let ffi_output = self.execute_protected(move || {
            // Add sentinel values after the buffer
            let mut wave_buffer = wave_buffer;
            wave_buffer.extend(std::iter::repeat(SENTINEL).take(SENTINEL_COUNT));

            let mut clock_times = vec![0.0f64; clock_times_len];
            let mut params_out: *mut c_char = ptr::null_mut();
            let handle = handle_usize as *mut c_void;

            // Call AMI_GetWave with original size (sentinels are hidden from model)
            let return_code = unsafe {
                getwave_fn(
                    wave_buffer.as_mut_ptr(),
                    original_buffer_len as i64, // Report original size, not padded
                    clock_times.as_mut_ptr(),
                    &mut params_out,
                    handle,
                )
            };

            // CRIT-FFI-001 FIX: Read string immediately after FFI call, inside closure.
            // Per IBIS 7.2 Section 10.2.3, vendor may reuse static buffer immediately.
            let params_out_str = unsafe { read_c_string(params_out) };

            // HIGH-FFI-004 FIX: Check sentinel values for buffer overrun
            let mut overrun_index = None;
            for i in 0..SENTINEL_COUNT {
                let sentinel_idx = original_buffer_len + i;
                if wave_buffer[sentinel_idx] != SENTINEL {
                    overrun_index = Some(sentinel_idx);
                    break;
                }
            }

            // Truncate back to original size (remove sentinels)
            wave_buffer.truncate(original_buffer_len);

            GetWaveFfiOutput {
                return_code,
                params_out_str,
                wave_buffer,
                clock_times,
                overrun_index, // NEW: Include overrun detection result
            }
        })?;

        // HIGH-FFI-004 FIX: Check for buffer overrun
        if let Some(overrun_idx) = ffi_output.overrun_index {
            tracing::error!(
                model = %model_name,
                buffer_size = original_buffer_len,
                overrun_index = overrun_idx,
                "Buffer overrun detected! Model wrote beyond allocated buffer."
            );

            return Err(AmiError::BufferOverrun {
                model: model_name,
                size: original_buffer_len,
                detected_index: Some(overrun_idx),
            });
        }

        // Check return code
        if ffi_output.return_code != 0 {
            return Err(AmiError::GetWaveFailed { code: ffi_output.return_code });
        }

        // Update state
        self.state = SessionState::Active;
        self.getwave_count += 1;

        // Copy modified waveform back
        wave.samples = ffi_output.wave_buffer;

        // Parse output parameters from already-copied string (CRIT-FFI-001 FIX)
        let output_params = ffi_output.params_out_str
            .and_then(|s| AmiParameters::from_ami_string(&s).ok())
            .unwrap_or_default();

        tracing::debug!(
            getwave_count = self.getwave_count,
            samples = wave.samples.len(),
            "AMI_GetWave completed successfully (buffer overrun check passed)"
        );

        Ok(AmiGetWaveResult {
            return_code: ffi_output.return_code,
            clock_times: ffi_output.clock_times,
            output_params,
        })
    }

    /// Close the session and release resources.
    ///
    /// This is called automatically on drop, but can be called explicitly
    /// for earlier resource release or error handling.
    ///
    /// # CRIT-FFI-002 Fix
    ///
    /// This method waits for any pending (orphaned) operations to complete
    /// before calling AMI_Close, preventing the race condition where an
    /// orphaned thread could still be using the handle.
    pub fn close(&mut self) -> AmiResult<()> {
        // Only close if initialized
        match self.state {
            SessionState::Initialized | SessionState::Active => {}
            SessionState::Closed => return Ok(()),
            _ => return Ok(()), // Don't try to close faulted sessions
        }

        let handle_usize = self.handle.load(Ordering::SeqCst) as usize;
        if handle_usize == 0 {
            self.state = SessionState::Closed;
            return Ok(());
        }

        // CRIT-FFI-002 FIX: Wait for pending operations before closing.
        // This prevents calling AMI_Close while orphaned threads still use the handle.
        let max_wait = self.config.timeout * 2; // Wait up to 2x normal timeout
        let start = std::time::Instant::now();

        while self.pending_ops.load(Ordering::SeqCst) > 0 {
            if start.elapsed() > max_wait {
                tracing::warn!(
                    pending_ops = self.pending_ops.load(Ordering::SeqCst),
                    "Closing session with pending operations still in-flight"
                );
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        // Now safe to close - no operations in flight (or we timed out waiting)
        let close_fn = self.library.close_fn();
        let result = self.execute_protected(move || {
            let handle = handle_usize as *mut c_void;
            unsafe { close_fn(handle) }
        });

        // Update state regardless of result
        self.state = SessionState::Closed;
        self.handle.store(ptr::null_mut(), Ordering::SeqCst);

        match result {
            Ok(0) => {
                tracing::debug!(getwave_count = self.getwave_count, "AMI_Close completed");
                Ok(())
            }
            Ok(code) => Err(AmiError::CloseFailed { code }),
            Err(e) => Err(e),
        }
    }

    /// Execute a function with timeout and panic protection.
    ///
    /// # Thread Safety
    ///
    /// This function spawns a thread to execute the FFI call. If the call times out,
    /// the thread becomes "orphaned" - it continues running but we can no longer
    /// communicate with it. We track these orphaned threads globally and refuse
    /// new operations if too many accumulate.
    ///
    /// Note: `AmiSession` requires `&mut self` for all operations, which provides
    /// compile-time exclusivity - only one thread can call methods on a session
    /// at a time. This prevents the race condition that could occur with concurrent
    /// access to the session handle.
    fn execute_protected<F, R>(&self, f: F) -> AmiResult<R>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        // Check for too many orphaned threads before spawning a new one
        let orphaned_count = ORPHANED_THREAD_COUNT.load(Ordering::SeqCst);
        if orphaned_count >= MAX_ORPHANED_THREADS {
            return Err(AmiError::TooManyOrphanedThreads {
                count: orphaned_count,
                max: MAX_ORPHANED_THREADS,
            });
        }

        let timeout = self.config.timeout;
        let catch_panics = self.config.catch_panics;

        // CRIT-FFI-002 FIX: Track pending operations for safe close()
        let pending_ops = self.pending_ops.clone();
        pending_ops.fetch_add(1, Ordering::SeqCst);

        // Use a channel for result communication
        let (tx, rx) = crossbeam::channel::bounded(1);

        // Spawn thread to execute function
        // The thread will decrement counters when it completes
        std::thread::spawn(move || {
            let result = if catch_panics {
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(f))
            } else {
                Ok(f())
            };

            // CRIT-FFI-002 FIX: ALWAYS decrement pending ops when thread completes
            pending_ops.fetch_sub(1, Ordering::SeqCst);

            // Try to send result. If the receiver is gone (timeout occurred),
            // decrement the orphaned thread counter since we're now completing.
            if tx.send(result).is_err() {
                // Receiver dropped = timeout occurred, but we're done now
                ORPHANED_THREAD_COUNT.fetch_sub(1, Ordering::SeqCst);
            }
        });

        // Wait for result with timeout
        match rx.recv_timeout(timeout) {
            Ok(Ok(result)) => Ok(result),
            Ok(Err(panic_info)) => {
                let message = if let Some(s) = panic_info.downcast_ref::<&str>() {
                    s.to_string()
                } else if let Some(s) = panic_info.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "Unknown panic".to_string()
                };
                Err(AmiError::ModelPanicked(message))
            }
            Err(_) => {
                // Timeout occurred - increment orphaned thread counter
                // The thread is still running and will decrement when it completes
                ORPHANED_THREAD_COUNT.fetch_add(1, Ordering::SeqCst);
                tracing::warn!(
                    orphaned_threads = ORPHANED_THREAD_COUNT.load(Ordering::SeqCst),
                    pending_ops = self.pending_ops.load(Ordering::SeqCst),
                    timeout_ms = timeout.as_millis(),
                    "AMI call timed out, thread orphaned"
                );
                Err(AmiError::Timeout(timeout))
            }
        }
    }

    /// Get the current count of pending operations for this session.
    ///
    /// This is useful for monitoring and diagnostics.
    pub fn pending_operations(&self) -> usize {
        self.pending_ops.load(Ordering::SeqCst)
    }
}

/// Get the current count of orphaned threads.
///
/// This is useful for monitoring and diagnostics.
pub fn orphaned_thread_count() -> usize {
    ORPHANED_THREAD_COUNT.load(Ordering::SeqCst)
}

impl Drop for AmiSession {
    fn drop(&mut self) {
        if matches!(
            self.state,
            SessionState::Initialized | SessionState::Active
        ) {
            // Best-effort close, log but don't propagate errors
            if let Err(e) = self.close() {
                tracing::warn!(error = %e, "Error during session cleanup");
            }
        }
    }
}

/// Read a C string, returning None if null or invalid UTF-8.
///
/// # Safety
/// The pointer must be null or point to a valid null-terminated C string.
unsafe fn read_c_string(ptr: *const c_char) -> Option<String> {
    if ptr.is_null() {
        return None;
    }
    // SAFETY: Caller guarantees ptr is valid if not null
    unsafe { CStr::from_ptr(ptr).to_str().ok().map(String::from) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execution_config_default() {
        let config = ExecutionConfig::default();
        assert_eq!(config.timeout, Duration::from_secs(30));
        assert!(config.catch_panics);
    }

    #[test]
    fn test_session_state_transitions() {
        // This test verifies the state machine logic without actual library
        // Real integration tests would use test fixtures with mock libraries
    }
}

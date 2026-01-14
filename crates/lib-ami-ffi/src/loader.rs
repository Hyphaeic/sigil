//! Dynamic library loading for AMI models.
//!
//! This module handles loading vendor-supplied shared libraries and
//! extracting the required AMI function symbols.

use crate::error::{AmiError, AmiResult};
use libloading::{Library, Symbol};
use std::ffi::{c_char, c_double, c_void};
use std::os::raw::c_long;
use std::path::Path;
use std::sync::Arc;

/// Function signature for AMI_Init.
///
/// ```c
/// long AMI_Init(
///     double *impulse_matrix,
///     long    row_size,
///     long    aggressors,
///     double  sample_interval,
///     double  bit_time,
///     char   *AMI_parameters_in,
///     char  **AMI_parameters_out,
///     void  **AMI_memory_handle,
///     char  **msg
/// );
/// ```
pub type AmiInitFn = unsafe extern "C" fn(
    impulse_matrix: *mut c_double,
    row_size: c_long,
    aggressors: c_long,
    sample_interval: c_double,
    bit_time: c_double,
    ami_parameters_in: *const c_char,
    ami_parameters_out: *mut *mut c_char,
    ami_memory_handle: *mut *mut c_void,
    msg: *mut *mut c_char,
) -> c_long;

/// Function signature for AMI_GetWave.
///
/// ```c
/// long AMI_GetWave(
///     double *wave,
///     long    wave_size,
///     double *clock_times,
///     char  **AMI_parameters_out,
///     void   *AMI_memory_handle
/// );
/// ```
pub type AmiGetWaveFn = unsafe extern "C" fn(
    wave: *mut c_double,
    wave_size: c_long,
    clock_times: *mut c_double,
    ami_parameters_out: *mut *mut c_char,
    ami_memory_handle: *mut c_void,
) -> c_long;

/// Function signature for AMI_Close.
///
/// ```c
/// long AMI_Close(void *AMI_memory_handle);
/// ```
pub type AmiCloseFn = unsafe extern "C" fn(ami_memory_handle: *mut c_void) -> c_long;

/// Function signature for AMI_Free (optional).
///
/// ```c
/// void AMI_Free(void *ptr);
/// ```
///
/// Per IBIS 7.2 Section 10.2.2/10.2.3, the simulator must call AMI_Free
/// to release memory allocated by AMI_Init (msg, AMI_parameters_out) and
/// AMI_GetWave (AMI_parameters_out).
///
/// Note: AMI_Free is optional in IBIS 7.2 Section 10.2.1 for backwards
/// compatibility with legacy models. If not present, memory leaks will occur.
pub type AmiFreeFn = unsafe extern "C" fn(ptr: *mut c_void);

/// Loaded AMI library with extracted function pointers.
pub struct AmiLibrary {
    /// The underlying dynamic library handle.
    #[allow(dead_code)]
    library: Library,

    /// Path to the library file.
    pub path: String,

    /// AMI_Init function pointer.
    ami_init: AmiInitFn,

    /// AMI_GetWave function pointer (optional in some models).
    ami_getwave: Option<AmiGetWaveFn>,

    /// AMI_Close function pointer.
    ami_close: AmiCloseFn,

    /// AMI_Free function pointer (optional, for memory cleanup).
    ///
    /// CRIT-NEW-001 FIX: Per IBIS 7.2 §10.2.2/§10.2.3, the simulator must
    /// call AMI_Free to release vendor-allocated strings (msg, AMI_parameters_out).
    ///
    /// If not present, memory will leak on every Init/GetWave call. This is
    /// tolerable for legacy models but will cause OOM on long BER runs.
    ami_free: Option<AmiFreeFn>,
}

impl AmiLibrary {
    /// Load an AMI model from a shared library file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the `.dll` (Windows) or `.so` (Linux) file
    ///
    /// # Safety
    ///
    /// The library must contain valid AMI function implementations.
    /// Invalid or malicious libraries may cause undefined behavior.
    pub fn load<P: AsRef<Path>>(path: P) -> AmiResult<Arc<Self>> {
        let path = path.as_ref();
        let path_str = path.display().to_string();

        // Load the library (libloading will handle file-not-found errors)
        let library = unsafe { Library::new(path) }
            .map_err(|e| AmiError::load_error(&path_str, e))?;

        // Extract required symbols
        let ami_init: AmiInitFn = unsafe {
            *library
                .get::<AmiInitFn>(b"AMI_Init\0")
                .map_err(|_| AmiError::symbol_not_found("AMI_Init"))?
        };

        let ami_close: AmiCloseFn = unsafe {
            *library
                .get::<AmiCloseFn>(b"AMI_Close\0")
                .map_err(|_| AmiError::symbol_not_found("AMI_Close"))?
        };

        // AMI_GetWave is optional
        let ami_getwave: Option<AmiGetWaveFn> = unsafe {
            library
                .get::<AmiGetWaveFn>(b"AMI_GetWave\0")
                .ok()
                .map(|s| *s)
        };

        // CRIT-NEW-001 FIX: Try to load AMI_Free (optional per IBIS 7.2 §10.2.1)
        let ami_free: Option<AmiFreeFn> = unsafe {
            library
                .get::<AmiFreeFn>(b"AMI_Free\0")
                .ok()
                .map(|s| *s)
        };

        if ami_free.is_none() {
            tracing::warn!(
                path = %path_str,
                "AMI_Free not found — memory leaks possible (legacy model?)"
            );
        }

        tracing::info!(
            path = %path_str,
            has_getwave = ami_getwave.is_some(),
            has_free = ami_free.is_some(),
            "Loaded AMI library"
        );

        Ok(Arc::new(Self {
            library,
            path: path_str,
            ami_init,
            ami_getwave,
            ami_close,
            ami_free,
        }))
    }

    /// Check if this library supports GetWave.
    pub fn supports_getwave(&self) -> bool {
        self.ami_getwave.is_some()
    }

    /// Get the AMI_Init function pointer.
    pub fn init_fn(&self) -> AmiInitFn {
        self.ami_init
    }

    /// Get the AMI_GetWave function pointer, if available.
    pub fn getwave_fn(&self) -> Option<AmiGetWaveFn> {
        self.ami_getwave
    }

    /// Get the AMI_Close function pointer.
    pub fn close_fn(&self) -> AmiCloseFn {
        self.ami_close
    }

    /// Free vendor-allocated memory (CRIT-NEW-001 FIX).
    ///
    /// Per IBIS 7.2 §10.2.2/§10.2.3, the simulator must call AMI_Free
    /// to release strings allocated by AMI_Init and AMI_GetWave.
    ///
    /// # Safety
    ///
    /// - `ptr` must be a valid pointer returned by the vendor model
    /// - Must not be called twice on the same pointer (double-free)
    /// - If AMI_Free is not available (legacy model), this is a no-op
    ///   and memory will leak (unavoidable for old models)
    ///
    /// # Arguments
    ///
    /// * `ptr` - Pointer to vendor-allocated memory (typically *mut c_char)
    pub fn free_vendor_memory(&self, ptr: *mut c_char) {
        if ptr.is_null() {
            return;
        }

        if let Some(ami_free) = self.ami_free {
            unsafe {
                ami_free(ptr as *mut c_void);
            }
            tracing::trace!("AMI_Free called on vendor string");
        } else {
            // Legacy model without AMI_Free — leak is unavoidable
            tracing::trace!("Skipping AMI_Free (not available in model) — memory leak");
        }
    }
}

// HIGH-FFI-003 FIX: Thread Safety Documentation
//
// IMPORTANT: While AmiLibrary itself is Send + Sync (it only stores function
// pointers and a Library handle), the IBIS 7.2 specification (Section 10.1)
// explicitly states:
//
//   "The model may maintain internal state between AMI_Init and AMI_Close.
//    The simulator shall not call the same model instance concurrently
//    from multiple threads."
//
// This means:
// - It is safe to LOAD the same .dll/.so from multiple threads
// - It is NOT safe to call AMI_* functions on the same session from
//   multiple threads
// - Each AmiSession should be confined to a single thread, or access
//   should be serialized via Mutex
//
// The AmiSession type in lifecycle.rs does NOT implement Sync, which
// prevents sharing a session across threads. If you need parallel
// simulation, create separate AmiSession instances (each with its own
// AMI_Init call and handle).
//
// SAFETY: AmiLibrary stores only function pointers and the dlopen handle.
// These are inherently thread-safe to store and copy. The thread-safety
// concern is about CALLING the functions, which is handled at the
// AmiSession level.
unsafe impl Send for AmiLibrary {}
unsafe impl Sync for AmiLibrary {}

/// Information about a loaded AMI library.
#[derive(Clone, Debug)]
pub struct LibraryInfo {
    /// Path to the library.
    pub path: String,

    /// Whether AMI_GetWave is available.
    pub has_getwave: bool,

    /// Platform-specific library format.
    pub format: LibraryFormat,
}

/// Platform-specific library format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LibraryFormat {
    /// Windows DLL.
    Dll,
    /// Linux/Unix shared object.
    So,
    /// macOS dynamic library.
    Dylib,
    /// Unknown format.
    Unknown,
}

impl LibraryFormat {
    /// Detect format from file extension.
    pub fn from_path<P: AsRef<Path>>(path: P) -> Self {
        match path.as_ref().extension().and_then(|e| e.to_str()) {
            Some("dll") | Some("DLL") => Self::Dll,
            Some("so") => Self::So,
            Some("dylib") => Self::Dylib,
            _ => Self::Unknown,
        }
    }

    /// Get the default extension for the current platform.
    #[cfg(target_os = "windows")]
    pub fn native() -> Self {
        Self::Dll
    }

    #[cfg(target_os = "linux")]
    pub fn native() -> Self {
        Self::So
    }

    #[cfg(target_os = "macos")]
    pub fn native() -> Self {
        Self::Dylib
    }

    #[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
    pub fn native() -> Self {
        Self::Unknown
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_library_format_detection() {
        assert_eq!(LibraryFormat::from_path("model.dll"), LibraryFormat::Dll);
        assert_eq!(LibraryFormat::from_path("libmodel.so"), LibraryFormat::So);
        assert_eq!(
            LibraryFormat::from_path("libmodel.dylib"),
            LibraryFormat::Dylib
        );
        assert_eq!(LibraryFormat::from_path("model.txt"), LibraryFormat::Unknown);
    }
}

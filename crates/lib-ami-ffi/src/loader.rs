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

        tracing::info!(
            path = %path_str,
            has_getwave = ami_getwave.is_some(),
            "Loaded AMI library"
        );

        Ok(Arc::new(Self {
            library,
            path: path_str,
            ami_init,
            ami_getwave,
            ami_close,
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
}

// AmiLibrary is Send + Sync because we only store function pointers
// and the Library handle, which are thread-safe to access.
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

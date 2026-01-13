//! Back-channel communication for Tx-Rx adaptation.
//!
//! In PCIe Gen 5 and beyond, the receiver can communicate back to the
//! transmitter to request equalization changes. This module provides
//! the message bus infrastructure for this communication.

use lib_types::ami::{AmiValue, BackChannelMessage};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Mutex, MutexGuard, PoisonError};

/// Helper trait to recover from poisoned mutexes.
///
/// If a thread panicked while holding a mutex, the mutex becomes "poisoned".
/// For the back-channel bus, we recover by accessing the data anyway, since
/// it's better to continue with potentially inconsistent data than to panic.
trait RecoverMutex<T> {
    fn lock_recover(&self) -> MutexGuard<'_, T>;
}

impl<T> RecoverMutex<T> for Mutex<T> {
    fn lock_recover(&self) -> MutexGuard<'_, T> {
        self.lock().unwrap_or_else(|poisoned| {
            tracing::warn!("Mutex was poisoned, recovering data");
            poisoned.into_inner()
        })
    }
}

pub use lib_types::ami::TrainingState;

/// Best training result.
///
/// # MED-TRAIN-002 Fix
///
/// This struct holds both FOM and preset together to ensure atomic updates.
/// Previously, best_fom and best_preset were separate mutexes, creating a
/// race condition where they could become inconsistent.
#[derive(Clone, Copy, Debug)]
struct BestResult {
    fom: f64,
    preset: u8,
}

/// Message bus for back-channel communication between Tx and Rx models.
///
/// The bus provides bidirectional communication channels:
/// - Rx → Tx: Receiver sends adaptation requests
/// - Tx → Rx: Transmitter acknowledges changes
///
/// This is used during link training and continuous adaptation.
pub struct BackChannelBus {
    /// Messages from Rx to Tx.
    rx_to_tx: Mutex<VecDeque<BackChannelMessage>>,

    /// Messages from Tx to Rx.
    tx_to_rx: Mutex<VecDeque<BackChannelMessage>>,

    /// Current training state.
    state: AtomicU8,

    /// Best result seen during training (FOM + preset atomically consistent).
    ///
    /// # MED-TRAIN-002 Fix
    ///
    /// Previously FOM and preset were separate mutexes, which could lead to
    /// mismatched values in parallel training scenarios. Now both values are
    /// protected by a single mutex for atomic consistency.
    best_result: Mutex<Option<BestResult>>,
}

impl BackChannelBus {
    /// Create a new back-channel bus.
    pub fn new() -> Self {
        Self {
            rx_to_tx: Mutex::new(VecDeque::new()),
            tx_to_rx: Mutex::new(VecDeque::new()),
            state: AtomicU8::new(TrainingState::Idle as u8),
            best_result: Mutex::new(None), // MED-TRAIN-002: Atomic FOM+preset
        }
    }

    /// Get the current training state.
    ///
    /// # HIGH-TRAIN-001 Fix
    ///
    /// Previously this function silently fell back to Idle on unknown states,
    /// which could cause unexpected training restarts if a new state was added.
    /// Per PCIe 5.0 Section 4.2.6.3, state machine errors must be explicit.
    ///
    /// # Panics
    ///
    /// Panics if an unknown training state value is encountered. This is a
    /// programming error (state was set to invalid value) and should be caught
    /// during testing, not silently ignored in production.
    pub fn state(&self) -> TrainingState {
        let raw_state = self.state.load(Ordering::SeqCst);
        match raw_state {
            0 => TrainingState::Idle,
            1 => TrainingState::PresetSweep,
            2 => TrainingState::CoarseAdaptation,
            3 => TrainingState::FineAdaptation,
            4 => TrainingState::Converged,
            5 => TrainingState::Failed,
            _ => {
                tracing::error!(
                    raw_state,
                    "Unknown training state encountered! This is a programming error."
                );
                panic!(
                    "Invalid training state: {}. Valid states are 0-5 (Idle through Failed).",
                    raw_state
                );
            }
        }
    }

    /// Set the training state.
    pub fn set_state(&self, state: TrainingState) {
        self.state.store(state as u8, Ordering::SeqCst);
    }

    /// Send a message from Rx to Tx.
    pub fn rx_send(&self, msg: BackChannelMessage) {
        self.rx_to_tx.lock_recover().push_back(msg);
    }

    /// Receive a message at Tx (from Rx).
    pub fn tx_receive(&self) -> Option<BackChannelMessage> {
        self.rx_to_tx.lock_recover().pop_front()
    }

    /// Peek at the next message for Tx without removing it.
    pub fn tx_peek(&self) -> Option<BackChannelMessage> {
        self.rx_to_tx.lock_recover().front().cloned()
    }

    /// Send a message from Tx to Rx.
    pub fn tx_send(&self, msg: BackChannelMessage) {
        self.tx_to_rx.lock_recover().push_back(msg);
    }

    /// Receive a message at Rx (from Tx).
    pub fn rx_receive(&self) -> Option<BackChannelMessage> {
        self.tx_to_rx.lock_recover().pop_front()
    }

    /// Peek at the next message for Rx without removing it.
    pub fn rx_peek(&self) -> Option<BackChannelMessage> {
        self.tx_to_rx.lock_recover().front().cloned()
    }

    /// Check if there are pending messages for Tx.
    pub fn has_tx_messages(&self) -> bool {
        !self.rx_to_tx.lock_recover().is_empty()
    }

    /// Check if there are pending messages for Rx.
    pub fn has_rx_messages(&self) -> bool {
        !self.tx_to_rx.lock_recover().is_empty()
    }

    /// Clear all pending messages.
    pub fn clear(&self) {
        self.rx_to_tx.lock_recover().clear();
        self.tx_to_rx.lock_recover().clear();
    }

    /// Record a figure of merit observation.
    ///
    /// # MED-TRAIN-002 Fix
    ///
    /// Now uses a single mutex to atomically update both FOM and preset,
    /// preventing race conditions in parallel training scenarios.
    pub fn record_fom(&self, fom: f64, preset: u8) {
        let mut best = self.best_result.lock_recover();

        if best.is_none() || fom > best.unwrap().fom {
            *best = Some(BestResult { fom, preset });
            tracing::debug!(fom, preset, "New best training result recorded");
        }
    }

    /// Get the best figure of merit and associated preset.
    ///
    /// # MED-TRAIN-002 Fix
    ///
    /// Returns an atomically consistent (FOM, preset) pair from a single mutex.
    pub fn best_result(&self) -> Option<(f64, u8)> {
        self.best_result.lock_recover()
            .map(|result| (result.fom, result.preset))
    }

    /// Reset training state and results.
    pub fn reset(&self) {
        self.clear();
        self.set_state(TrainingState::Idle);
        *self.best_result.lock_recover() = None;
    }
}

impl Default for BackChannelBus {
    fn default() -> Self {
        Self::new()
    }
}

/// Training configuration.
#[derive(Clone, Debug)]
pub struct TrainingConfig {
    /// Maximum iterations for preset sweep.
    pub max_preset_sweep: usize,

    /// Maximum iterations for coarse adaptation.
    pub max_coarse_iterations: usize,

    /// Maximum iterations for fine adaptation.
    pub max_fine_iterations: usize,

    /// Convergence threshold for adaptation.
    pub convergence_threshold: f64,

    /// Number of bits per adaptation chunk.
    pub bits_per_chunk: u64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            max_preset_sweep: 11, // P0-P10
            max_coarse_iterations: 10,
            max_fine_iterations: 50,
            convergence_threshold: 0.01,
            bits_per_chunk: 1_000_000,
        }
    }
}

/// Result from link training.
#[derive(Clone, Debug)]
pub struct TrainingResult {
    /// Final training state.
    pub final_state: TrainingState,

    /// Best Tx preset found.
    pub best_preset: Option<u8>,

    /// Best figure of merit achieved.
    pub best_fom: Option<f64>,

    /// Final Tx coefficients.
    pub tx_coefficients: Option<TxCoefficients>,

    /// Final Rx equalization settings.
    pub rx_settings: Option<RxSettings>,

    /// Number of training iterations.
    pub iterations: usize,
}

/// Tx equalization coefficients.
#[derive(Clone, Copy, Debug, Default)]
pub struct TxCoefficients {
    /// Pre-cursor tap (typically negative).
    pub pre_cursor: f64,

    /// Main cursor (typically positive).
    pub main_cursor: f64,

    /// Post-cursor tap (typically negative).
    pub post_cursor: f64,
}

impl TxCoefficients {
    /// Create from normalized values.
    pub fn new(pre: f64, main: f64, post: f64) -> Self {
        Self {
            pre_cursor: pre,
            main_cursor: main,
            post_cursor: post,
        }
    }

    /// Sum of absolute tap values (should be <= 1 for valid config).
    pub fn tap_sum(&self) -> f64 {
        self.pre_cursor.abs() + self.main_cursor + self.post_cursor.abs()
    }

    /// Validate coefficient constraints.
    pub fn is_valid(&self) -> bool {
        self.main_cursor > 0.0
            && self.tap_sum() <= 1.0 + 1e-6
            && self.pre_cursor <= 0.0
            && self.post_cursor <= 0.0
    }
}

/// Rx equalization settings.
#[derive(Clone, Debug, Default)]
pub struct RxSettings {
    /// CTLE DC gain (dB).
    pub ctle_dc_gain: f64,

    /// CTLE peaking frequency (GHz).
    pub ctle_peak_freq: f64,

    /// CTLE peaking gain (dB).
    pub ctle_peak_gain: f64,

    /// DFE tap values.
    pub dfe_taps: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backchannel_bus_roundtrip() {
        let bus = BackChannelBus::new();

        // Rx sends to Tx
        bus.rx_send(BackChannelMessage::PresetRequest { preset: 5 });
        assert!(bus.has_tx_messages());

        // Tx receives
        let msg = bus.tx_receive().unwrap();
        assert!(matches!(msg, BackChannelMessage::PresetRequest { preset: 5 }));
        assert!(!bus.has_tx_messages());

        // Tx sends to Rx
        bus.tx_send(BackChannelMessage::FigureOfMerit { value: 0.85 });
        assert!(bus.has_rx_messages());

        // Rx receives
        let msg = bus.rx_receive().unwrap();
        assert!(matches!(msg, BackChannelMessage::FigureOfMerit { value } if (value - 0.85).abs() < 1e-10));
    }

    #[test]
    fn test_training_state_transitions() {
        let bus = BackChannelBus::new();

        assert_eq!(bus.state(), TrainingState::Idle);

        bus.set_state(TrainingState::PresetSweep);
        assert_eq!(bus.state(), TrainingState::PresetSweep);

        bus.set_state(TrainingState::Converged);
        assert_eq!(bus.state(), TrainingState::Converged);
    }

    #[test]
    fn test_fom_tracking() {
        let bus = BackChannelBus::new();

        bus.record_fom(0.5, 0);
        bus.record_fom(0.8, 5);
        bus.record_fom(0.6, 10);

        let (best_fom, best_preset) = bus.best_result().unwrap();
        assert!((best_fom - 0.8).abs() < 1e-10);
        assert_eq!(best_preset, 5);
    }

    #[test]
    fn test_tx_coefficients_validation() {
        let valid = TxCoefficients::new(-0.1, 0.8, -0.1);
        assert!(valid.is_valid());

        let invalid = TxCoefficients::new(-0.5, 0.8, -0.5);
        assert!(!invalid.is_valid());
    }
}

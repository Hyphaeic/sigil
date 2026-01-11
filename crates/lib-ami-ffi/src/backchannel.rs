//! Back-channel communication for Tx-Rx adaptation.
//!
//! In PCIe Gen 5 and beyond, the receiver can communicate back to the
//! transmitter to request equalization changes. This module provides
//! the message bus infrastructure for this communication.

use lib_types::ami::{AmiValue, BackChannelMessage};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Mutex;

pub use lib_types::ami::TrainingState;

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

    /// Best figure of merit seen during training.
    best_fom: Mutex<Option<f64>>,

    /// Preset associated with best FOM.
    best_preset: Mutex<Option<u8>>,
}

impl BackChannelBus {
    /// Create a new back-channel bus.
    pub fn new() -> Self {
        Self {
            rx_to_tx: Mutex::new(VecDeque::new()),
            tx_to_rx: Mutex::new(VecDeque::new()),
            state: AtomicU8::new(TrainingState::Idle as u8),
            best_fom: Mutex::new(None),
            best_preset: Mutex::new(None),
        }
    }

    /// Get the current training state.
    pub fn state(&self) -> TrainingState {
        match self.state.load(Ordering::SeqCst) {
            0 => TrainingState::Idle,
            1 => TrainingState::PresetSweep,
            2 => TrainingState::CoarseAdaptation,
            3 => TrainingState::FineAdaptation,
            4 => TrainingState::Converged,
            5 => TrainingState::Failed,
            _ => TrainingState::Idle,
        }
    }

    /// Set the training state.
    pub fn set_state(&self, state: TrainingState) {
        self.state.store(state as u8, Ordering::SeqCst);
    }

    /// Send a message from Rx to Tx.
    pub fn rx_send(&self, msg: BackChannelMessage) {
        self.rx_to_tx.lock().unwrap().push_back(msg);
    }

    /// Receive a message at Tx (from Rx).
    pub fn tx_receive(&self) -> Option<BackChannelMessage> {
        self.rx_to_tx.lock().unwrap().pop_front()
    }

    /// Peek at the next message for Tx without removing it.
    pub fn tx_peek(&self) -> Option<BackChannelMessage> {
        self.rx_to_tx.lock().unwrap().front().cloned()
    }

    /// Send a message from Tx to Rx.
    pub fn tx_send(&self, msg: BackChannelMessage) {
        self.tx_to_rx.lock().unwrap().push_back(msg);
    }

    /// Receive a message at Rx (from Tx).
    pub fn rx_receive(&self) -> Option<BackChannelMessage> {
        self.tx_to_rx.lock().unwrap().pop_front()
    }

    /// Peek at the next message for Rx without removing it.
    pub fn rx_peek(&self) -> Option<BackChannelMessage> {
        self.tx_to_rx.lock().unwrap().front().cloned()
    }

    /// Check if there are pending messages for Tx.
    pub fn has_tx_messages(&self) -> bool {
        !self.rx_to_tx.lock().unwrap().is_empty()
    }

    /// Check if there are pending messages for Rx.
    pub fn has_rx_messages(&self) -> bool {
        !self.tx_to_rx.lock().unwrap().is_empty()
    }

    /// Clear all pending messages.
    pub fn clear(&self) {
        self.rx_to_tx.lock().unwrap().clear();
        self.tx_to_rx.lock().unwrap().clear();
    }

    /// Record a figure of merit observation.
    pub fn record_fom(&self, fom: f64, preset: u8) {
        let mut best = self.best_fom.lock().unwrap();
        let mut best_p = self.best_preset.lock().unwrap();

        if best.is_none() || fom > best.unwrap() {
            *best = Some(fom);
            *best_p = Some(preset);
        }
    }

    /// Get the best figure of merit and associated preset.
    pub fn best_result(&self) -> Option<(f64, u8)> {
        let fom = *self.best_fom.lock().unwrap();
        let preset = *self.best_preset.lock().unwrap();
        fom.zip(preset)
    }

    /// Reset training state and results.
    pub fn reset(&self) {
        self.clear();
        self.set_state(TrainingState::Idle);
        *self.best_fom.lock().unwrap() = None;
        *self.best_preset.lock().unwrap() = None;
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

//! Eye diagram analysis.

use lib_types::units::Seconds;
use lib_types::waveform::{EyeDiagram, StatisticalEye, Waveform};
use rayon::prelude::*;

/// Eye diagram analyzer for time-domain waveforms.
pub struct EyeAnalyzer {
    /// Samples per unit interval.
    samples_per_ui: usize,

    /// Number of voltage bins.
    voltage_bins: usize,

    /// Voltage range (min, max).
    voltage_range: (f64, f64),
}

impl EyeAnalyzer {
    /// Create a new eye analyzer.
    pub fn new(samples_per_ui: usize, voltage_bins: usize, voltage_range: (f64, f64)) -> Self {
        Self {
            samples_per_ui,
            voltage_bins,
            voltage_range,
        }
    }

    /// Analyze a waveform to produce an eye diagram.
    pub fn analyze(&self, waveform: &Waveform) -> EyeDiagram {
        let mut eye = EyeDiagram::new(self.samples_per_ui, self.voltage_bins, self.voltage_range);
        eye.accumulate(waveform, self.samples_per_ui);
        eye
    }

    /// Compute eye metrics.
    pub fn compute_metrics(&self, eye: &EyeDiagram) -> EyeMetrics {
        let height = eye.eye_height();

        // Compute width by finding zero crossings
        let center = self.samples_per_ui / 2;
        let threshold = self.voltage_bins / 2;

        let mut left_crossing = 0;
        let mut right_crossing = self.samples_per_ui;

        // Find left edge of eye
        for i in (0..center).rev() {
            let col = &eye.bins[i];
            let center_density: u64 = col[threshold.saturating_sub(5)..threshold + 5]
                .iter()
                .sum();
            if center_density > 0 {
                left_crossing = i;
                break;
            }
        }

        // Find right edge
        for i in center..self.samples_per_ui {
            let col = &eye.bins[i];
            let center_density: u64 = col[threshold.saturating_sub(5)..threshold + 5]
                .iter()
                .sum();
            if center_density > 0 {
                right_crossing = i;
                break;
            }
        }

        let width_samples = right_crossing - left_crossing;
        let width_ui = width_samples as f64 / self.samples_per_ui as f64;

        EyeMetrics {
            height,
            width_ui,
            jitter_rms: 0.0, // Would require CDR analysis
            snr: 0.0,        // Would require noise analysis
            ui_count: eye.ui_count,
        }
    }
}

/// Eye diagram metrics.
#[derive(Debug, Clone, Copy)]
pub struct EyeMetrics {
    /// Eye height in voltage units.
    pub height: f64,

    /// Eye width as fraction of UI.
    pub width_ui: f64,

    /// RMS jitter (if computed).
    pub jitter_rms: f64,

    /// Signal-to-noise ratio (if computed).
    pub snr: f64,

    /// Number of UI analyzed.
    pub ui_count: u64,
}

/// Statistical eye analyzer using superposition.
pub struct StatisticalEyeAnalyzer {
    /// Samples per unit interval.
    samples_per_ui: usize,
}

impl StatisticalEyeAnalyzer {
    /// Create a new statistical eye analyzer.
    pub fn new(samples_per_ui: usize) -> Self {
        Self { samples_per_ui }
    }

    /// Analyze a pulse response to compute statistical eye.
    pub fn analyze(&self, pulse_response: &Waveform) -> StatisticalEye {
        let num_ui = pulse_response.samples.len() / self.samples_per_ui;

        if num_ui < 2 {
            return StatisticalEye::new(self.samples_per_ui);
        }

        // Fold pulse into cursors at each phase
        let cursors: Vec<Vec<f64>> = (0..self.samples_per_ui)
            .into_par_iter()
            .map(|phase| {
                (0..num_ui)
                    .map(|ui| {
                        let idx = phase + ui * self.samples_per_ui;
                        pulse_response.samples.get(idx).copied().unwrap_or(0.0)
                    })
                    .collect()
            })
            .collect();

        // Find main cursor (largest absolute value)
        let main_cursor_ui = find_main_cursor(&cursors);

        // Compute worst-case eye
        let mut eye = StatisticalEye::new(self.samples_per_ui);

        for (phase, cursor_values) in cursors.iter().enumerate() {
            let main = cursor_values.get(main_cursor_ui).copied().unwrap_or(0.0);

            // Sum of absolute ISI contributions
            let pre_isi: f64 = cursor_values[..main_cursor_ui]
                .iter()
                .map(|v| v.abs())
                .sum();
            let post_isi: f64 = cursor_values[main_cursor_ui + 1..]
                .iter()
                .map(|v| v.abs())
                .sum();

            let total_isi = pre_isi + post_isi;

            // Worst-case high/low with ISI
            eye.high[phase] = main + total_isi;
            eye.low[phase] = main - total_isi;
        }

        eye
    }

    /// Compute eye margin at a given phase.
    pub fn margin_at_phase(&self, eye: &StatisticalEye, phase: usize) -> f64 {
        eye.high.get(phase).copied().unwrap_or(0.0)
            - eye.low.get(phase).copied().unwrap_or(0.0)
    }

    /// Compute overall figure of merit.
    pub fn figure_of_merit(&self, eye: &StatisticalEye) -> f64 {
        // Simple FOM: eye height * eye width
        let height = eye.eye_height();
        let width = eye.eye_width_ui();
        height * width
    }
}

/// Find the UI index with the main cursor (largest magnitude).
fn find_main_cursor(cursors: &[Vec<f64>]) -> usize {
    if cursors.is_empty() || cursors[0].is_empty() {
        return 0;
    }

    let num_ui = cursors[0].len();
    let center_phase = cursors.len() / 2;

    // Find UI with maximum absolute value at center phase
    cursors[center_phase]
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_statistical_eye_ideal_pulse() {
        // Ideal pulse (all energy in one UI)
        let samples_per_ui = 64;
        let mut samples = vec![0.0; samples_per_ui * 10];

        // Put ideal pulse in center UI
        for i in 0..samples_per_ui {
            samples[samples_per_ui * 5 + i] = 1.0;
        }

        let pulse = Waveform::new(samples, Seconds::from_ps(1.0), Seconds::ZERO);
        let analyzer = StatisticalEyeAnalyzer::new(samples_per_ui);
        let eye = analyzer.analyze(&pulse);

        // Ideal pulse should have eye height of 2.0 (symmetric ±1.0)
        assert!((eye.eye_height() - 2.0).abs() < 0.1);
        assert!((eye.eye_width_ui() - 1.0).abs() < 0.1);
    }
}

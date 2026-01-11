//! Time-domain waveform representation.
//!
//! Waveforms are the primary data structure for signal processing,
//! representing voltage (or normalized) signals sampled at uniform intervals.
//!
//! # Sample Semantics (LOW-006 FIX)
//!
//! Samples in a `Waveform` represent **point measurements** at discrete time instants.
//! For a waveform with `N` samples, the sample times are:
//!
//! ```text
//! t[i] = t_start + i * dt,  for i = 0, 1, ..., N-1
//! ```
//!
//! This means:
//! - `samples[0]` is measured at time `t_start`
//! - `samples[N-1]` is measured at time `t_start + (N-1) * dt`
//! - `t_end()` returns the time of the last sample, not the end of a sampling window
//! - `duration()` returns `N * dt`, which is the span from `t_start` to one sample *past* the last
//!
//! When performing operations like convolution or FFT, this convention should be
//! maintained consistently. For resampling, linear interpolation is used between
//! adjacent sample points.

use crate::units::{Seconds, Volts};
use serde::{Deserialize, Serialize};

/// A uniformly-sampled time-domain waveform.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Waveform {
    /// Sample values (voltage or normalized units).
    pub samples: Vec<f64>,

    /// Time step between consecutive samples.
    pub dt: Seconds,

    /// Time of the first sample (may be negative for acausal responses).
    pub t_start: Seconds,
}

impl Waveform {
    /// Create a new waveform from samples.
    pub fn new(samples: Vec<f64>, dt: Seconds, t_start: Seconds) -> Self {
        Self { samples, dt, t_start }
    }

    /// Create a zero-valued waveform of specified length.
    pub fn zeros(len: usize, dt: Seconds) -> Self {
        Self {
            samples: vec![0.0; len],
            dt,
            t_start: Seconds::ZERO,
        }
    }

    /// Number of samples in the waveform.
    #[inline]
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Check if the waveform is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Total duration of the waveform.
    #[inline]
    pub fn duration(&self) -> Seconds {
        Seconds(self.samples.len() as f64 * self.dt.0)
    }

    /// Time of the last sample.
    ///
    /// For a waveform with N samples at times t_start, t_start+dt, ..., t_start+(N-1)*dt,
    /// this returns t_start + (N-1)*dt, which is the time of the last sample.
    ///
    /// Returns t_start if the waveform is empty.
    #[inline]
    pub fn t_end(&self) -> Seconds {
        if self.samples.is_empty() {
            return self.t_start;
        }
        Seconds(self.t_start.0 + (self.samples.len() - 1) as f64 * self.dt.0)
    }

    /// Get the time value for a given sample index.
    #[inline]
    pub fn time_at(&self, index: usize) -> Seconds {
        Seconds(self.t_start.0 + index as f64 * self.dt.0)
    }

    /// Sample rate (reciprocal of dt).
    #[inline]
    pub fn sample_rate(&self) -> f64 {
        1.0 / self.dt.0
    }

    /// Find the index of the sample nearest to a given time.
    pub fn index_at_time(&self, t: Seconds) -> usize {
        let offset = (t.0 - self.t_start.0) / self.dt.0;
        offset.round().max(0.0).min((self.samples.len() - 1) as f64) as usize
    }

    /// Interpolate the waveform value at an arbitrary time.
    pub fn value_at(&self, t: Seconds) -> f64 {
        let offset = (t.0 - self.t_start.0) / self.dt.0;
        let idx = offset.floor() as isize;

        if idx < 0 {
            return self.samples.first().copied().unwrap_or(0.0);
        }
        if idx >= self.samples.len() as isize - 1 {
            return self.samples.last().copied().unwrap_or(0.0);
        }

        let idx = idx as usize;
        let frac = offset.fract();
        self.samples[idx] * (1.0 - frac) + self.samples[idx + 1] * frac
    }

    /// Peak-to-peak amplitude.
    pub fn peak_to_peak(&self) -> f64 {
        let (min, max) = self.samples.iter().fold((f64::MAX, f64::MIN), |(min, max), &v| {
            (min.min(v), max.max(v))
        });
        max - min
    }

    /// Maximum absolute value.
    pub fn max_abs(&self) -> f64 {
        self.samples.iter().map(|v| v.abs()).fold(0.0, f64::max)
    }

    /// Root mean square value.
    pub fn rms(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let sum_sq: f64 = self.samples.iter().map(|v| v * v).sum();
        (sum_sq / self.samples.len() as f64).sqrt()
    }

    /// Normalize to unit peak amplitude.
    pub fn normalize(&mut self) {
        let peak = self.max_abs();
        if peak > 0.0 {
            for sample in &mut self.samples {
                *sample /= peak;
            }
        }
    }

    /// Create a normalized copy.
    pub fn normalized(&self) -> Self {
        let mut copy = self.clone();
        copy.normalize();
        copy
    }

    /// Scale all samples by a factor.
    pub fn scale(&mut self, factor: f64) {
        for sample in &mut self.samples {
            *sample *= factor;
        }
    }

    /// Add a DC offset.
    pub fn add_offset(&mut self, offset: f64) {
        for sample in &mut self.samples {
            *sample += offset;
        }
    }

    /// Extract a time window from the waveform.
    pub fn window(&self, t_start: Seconds, t_end: Seconds) -> Self {
        let idx_start = self.index_at_time(t_start);
        let idx_end = self.index_at_time(t_end).min(self.samples.len() - 1);

        Self {
            samples: self.samples[idx_start..=idx_end].to_vec(),
            dt: self.dt,
            t_start,
        }
    }

    /// Resample to a new time step using linear interpolation.
    pub fn resample(&self, new_dt: Seconds) -> Self {
        let new_len = (self.duration().0 / new_dt.0).ceil() as usize;
        let mut new_samples = Vec::with_capacity(new_len);

        for i in 0..new_len {
            let t = Seconds(self.t_start.0 + i as f64 * new_dt.0);
            new_samples.push(self.value_at(t));
        }

        Self {
            samples: new_samples,
            dt: new_dt,
            t_start: self.t_start,
        }
    }

    /// Zero-pad the waveform to a specified length.
    pub fn zero_pad(&mut self, new_len: usize) {
        if new_len > self.samples.len() {
            self.samples.resize(new_len, 0.0);
        }
    }

    /// Trim leading/trailing zeros within a tolerance.
    ///
    /// Removes samples from the beginning and end of the waveform that have
    /// absolute values less than or equal to the tolerance. Updates t_start
    /// to reflect the new starting time.
    pub fn trim_zeros(&mut self, tolerance: f64) {
        if self.samples.is_empty() {
            return;
        }

        // Find first non-zero
        let start = self.samples.iter()
            .position(|&v| v.abs() > tolerance)
            .unwrap_or(self.samples.len());

        // Find last non-zero
        let end = self.samples.iter()
            .rposition(|&v| v.abs() > tolerance)
            .unwrap_or(0);

        // Check if any trimming is needed
        if start > 0 || end < self.samples.len().saturating_sub(1) {
            // Make sure we have a valid range
            if start <= end {
                self.t_start = Seconds(self.t_start.0 + start as f64 * self.dt.0);
                self.samples = self.samples[start..=end].to_vec();
            } else {
                // All samples are zeros - reduce to empty
                self.samples.clear();
            }
        }
    }
}

/// Waveform with explicit voltage units.
// LOW-003 FIX: Add Serialize/Deserialize to match inner Waveform
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VoltageWaveform {
    /// Underlying waveform data.
    pub inner: Waveform,
}

impl VoltageWaveform {
    /// Get voltage at a sample index.
    pub fn voltage_at_index(&self, index: usize) -> Volts {
        Volts(self.inner.samples.get(index).copied().unwrap_or(0.0))
    }

    /// Get voltage at a time.
    pub fn voltage_at_time(&self, t: Seconds) -> Volts {
        Volts(self.inner.value_at(t))
    }
}

impl From<Waveform> for VoltageWaveform {
    fn from(inner: Waveform) -> Self {
        Self { inner }
    }
}

/// Eye diagram computed from a waveform.
#[derive(Clone, Debug)]
pub struct EyeDiagram {
    /// Number of samples per UI.
    pub samples_per_ui: usize,

    /// 2D histogram bins [time_bin][voltage_bin] = count.
    pub bins: Vec<Vec<u64>>,

    /// Voltage range for binning.
    pub voltage_range: (f64, f64),

    /// Number of voltage bins.
    pub voltage_bins: usize,

    /// Total number of UI accumulated.
    pub ui_count: u64,
}

impl EyeDiagram {
    /// Create a new empty eye diagram.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `samples_per_ui` is 0
    /// - `voltage_bins` is 0
    /// - `voltage_range.0 >= voltage_range.1` (min >= max)
    pub fn new(samples_per_ui: usize, voltage_bins: usize, voltage_range: (f64, f64)) -> Self {
        // MED-008 FIX: Validate constructor parameters
        assert!(samples_per_ui > 0, "samples_per_ui must be > 0, got {}", samples_per_ui);
        assert!(voltage_bins > 0, "voltage_bins must be > 0, got {}", voltage_bins);
        assert!(
            voltage_range.0 < voltage_range.1,
            "voltage_range min ({}) must be less than max ({})",
            voltage_range.0,
            voltage_range.1
        );

        let bins = vec![vec![0u64; voltage_bins]; samples_per_ui];
        Self {
            samples_per_ui,
            bins,
            voltage_range,
            voltage_bins,
            ui_count: 0,
        }
    }

    /// Try to create a new empty eye diagram, returning an error for invalid parameters.
    pub fn try_new(
        samples_per_ui: usize,
        voltage_bins: usize,
        voltage_range: (f64, f64),
    ) -> Result<Self, &'static str> {
        if samples_per_ui == 0 {
            return Err("samples_per_ui must be > 0");
        }
        if voltage_bins == 0 {
            return Err("voltage_bins must be > 0");
        }
        if voltage_range.0 >= voltage_range.1 {
            return Err("voltage_range min must be less than max");
        }

        let bins = vec![vec![0u64; voltage_bins]; samples_per_ui];
        Ok(Self {
            samples_per_ui,
            bins,
            voltage_range,
            voltage_bins,
            ui_count: 0,
        })
    }

    /// Add a waveform to the eye diagram.
    pub fn accumulate(&mut self, waveform: &Waveform, samples_per_ui: usize) {
        let num_ui = waveform.samples.len() / samples_per_ui;

        for ui in 0..num_ui {
            for phase in 0..self.samples_per_ui {
                let idx = ui * samples_per_ui + phase;
                if idx >= waveform.samples.len() {
                    break;
                }

                let voltage = waveform.samples[idx];
                let v_norm = (voltage - self.voltage_range.0)
                    / (self.voltage_range.1 - self.voltage_range.0);
                let v_bin = (v_norm * self.voltage_bins as f64)
                    .floor()
                    .max(0.0)
                    .min((self.voltage_bins - 1) as f64) as usize;

                self.bins[phase][v_bin] += 1;
            }
            self.ui_count += 1;
        }
    }

    /// Eye height at the center crossing.
    pub fn eye_height(&self) -> f64 {
        let center_phase = self.samples_per_ui / 2;

        // Find the voltage bin with max count (should have two peaks for NRZ)
        let center_histogram = &self.bins[center_phase];

        // Simple approach: find gap between two peaks
        let threshold = *center_histogram.iter().max().unwrap_or(&1) / 4;

        let mut low_peak = 0;
        let mut high_peak = self.voltage_bins - 1;

        // Find low peak (first significant bin from bottom)
        for (i, &count) in center_histogram.iter().enumerate() {
            if count > threshold {
                low_peak = i;
                break;
            }
        }

        // Find high peak (first significant bin from top)
        for (i, &count) in center_histogram.iter().enumerate().rev() {
            if count > threshold {
                high_peak = i;
                break;
            }
        }

        // Eye height is the gap in voltage units
        let v_per_bin = (self.voltage_range.1 - self.voltage_range.0) / self.voltage_bins as f64;
        (high_peak - low_peak) as f64 * v_per_bin
    }
}

/// Statistical eye result from superposition analysis.
#[derive(Clone, Debug)]
pub struct StatisticalEye {
    /// High rail voltage at each phase.
    pub high: Vec<f64>,

    /// Low rail voltage at each phase.
    pub low: Vec<f64>,
}

impl StatisticalEye {
    /// Create a new statistical eye.
    pub fn new(samples_per_ui: usize) -> Self {
        Self {
            high: vec![0.0; samples_per_ui],
            low: vec![0.0; samples_per_ui],
        }
    }

    /// Eye height (minimum opening).
    pub fn eye_height(&self) -> f64 {
        self.high
            .iter()
            .zip(self.low.iter())
            .map(|(h, l)| h - l)
            .fold(f64::MAX, f64::min)
    }

    /// Eye width in UI (percentage of UI with positive opening).
    pub fn eye_width_ui(&self) -> f64 {
        let open_count = self
            .high
            .iter()
            .zip(self.low.iter())
            .filter(|(h, l)| *h > *l)
            .count();
        open_count as f64 / self.high.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_waveform_basics() {
        let samples = vec![0.0, 0.5, 1.0, 0.5, 0.0];
        let wf = Waveform::new(samples, Seconds::from_ps(10.0), Seconds::ZERO);

        assert_eq!(wf.len(), 5);
        assert!((wf.duration().as_ps() - 50.0).abs() < 0.01);
        assert!((wf.peak_to_peak() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_waveform_interpolation() {
        let samples = vec![0.0, 1.0, 0.0];
        let wf = Waveform::new(samples, Seconds(1.0), Seconds::ZERO);

        assert!((wf.value_at(Seconds(0.5)) - 0.5).abs() < 0.001);
        assert!((wf.value_at(Seconds(1.5)) - 0.5).abs() < 0.001);
    }
}

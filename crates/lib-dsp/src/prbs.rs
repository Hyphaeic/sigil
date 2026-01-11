//! Pseudo-Random Bit Sequence (PRBS) generation.

use lib_types::units::Seconds;
use lib_types::waveform::Waveform;

/// PRBS generator using linear feedback shift register.
pub struct PrbsGenerator {
    /// Current LFSR state.
    state: u64,

    /// Feedback taps (XOR mask).
    taps: u64,

    /// Sequence length (2^order - 1).
    length: u64,

    /// PRBS order.
    order: u8,

    /// Bits generated so far.
    count: u64,
}

impl PrbsGenerator {
    /// Create a new PRBS generator.
    ///
    /// Supported orders: 7, 9, 11, 15, 23, 31
    pub fn new(order: u8) -> Self {
        let (taps, length) = match order {
            7 => (0b1100000, (1u64 << 7) - 1),
            9 => (0b100010000, (1u64 << 9) - 1),
            11 => (0b10100000000, (1u64 << 11) - 1),
            15 => (0b110000000000000, (1u64 << 15) - 1),
            23 => (0b100001000000000000000, (1u64 << 23) - 1),
            31 => (0b1001000000000000000000000000000, (1u64 << 31) - 1),
            _ => panic!("Unsupported PRBS order: {order}. Use 7, 9, 11, 15, 23, or 31"),
        };

        Self {
            state: 1, // Non-zero initial state
            taps,
            length,
            order,
            count: 0,
        }
    }

    /// Get the PRBS order.
    pub fn order(&self) -> u8 {
        self.order
    }

    /// Get the sequence length.
    pub fn length(&self) -> u64 {
        self.length
    }

    /// Generate the next bit.
    pub fn next_bit(&mut self) -> u8 {
        // XOR all tapped bits
        let feedback = (self.state & self.taps).count_ones() & 1;

        // Shift and insert feedback
        self.state = (self.state >> 1) | ((feedback as u64) << (self.order - 1));

        self.count += 1;

        (self.state & 1) as u8
    }

    /// Generate N bits.
    pub fn generate_bits(&mut self, n: u64) -> Vec<u8> {
        (0..n).map(|_| self.next_bit()).collect()
    }

    /// Reset the generator to initial state.
    pub fn reset(&mut self) {
        self.state = 1;
        self.count = 0;
    }

    /// Set a custom initial state.
    pub fn set_state(&mut self, state: u64) {
        self.state = state & ((1 << self.order) - 1);
        if self.state == 0 {
            self.state = 1; // Avoid stuck state
        }
    }

    /// Get the current count of generated bits.
    pub fn bit_count(&self) -> u64 {
        self.count
    }

    /// Generate a voltage waveform from PRBS.
    ///
    /// # Arguments
    ///
    /// * `num_bits` - Number of bits to generate
    /// * `samples_per_bit` - Samples per bit (UI)
    /// * `dt` - Sample interval
    /// * `v_high` - Voltage for logic 1
    /// * `v_low` - Voltage for logic 0
    pub fn generate_waveform(
        &mut self,
        num_bits: u64,
        samples_per_bit: usize,
        dt: Seconds,
        v_high: f64,
        v_low: f64,
    ) -> Waveform {
        let total_samples = num_bits as usize * samples_per_bit;
        let mut samples = Vec::with_capacity(total_samples);

        for _ in 0..num_bits {
            let bit = self.next_bit();
            let voltage = if bit == 1 { v_high } else { v_low };

            for _ in 0..samples_per_bit {
                samples.push(voltage);
            }
        }

        Waveform {
            samples,
            dt,
            t_start: Seconds::ZERO,
        }
    }

    /// Generate a bipolar NRZ waveform (+1/-1).
    pub fn generate_nrz(&mut self, num_bits: u64, samples_per_bit: usize, dt: Seconds) -> Waveform {
        self.generate_waveform(num_bits, samples_per_bit, dt, 1.0, -1.0)
    }
}

/// Calculate run-length statistics for a PRBS.
pub fn run_length_stats(bits: &[u8]) -> RunLengthStats {
    let mut stats = RunLengthStats::default();

    if bits.is_empty() {
        return stats;
    }

    let mut current_run = 1usize;
    let mut current_bit = bits[0];

    for &bit in &bits[1..] {
        if bit == current_bit {
            current_run += 1;
        } else {
            // Record run
            if current_bit == 1 {
                stats.ones_runs.push(current_run);
            } else {
                stats.zeros_runs.push(current_run);
            }
            current_run = 1;
            current_bit = bit;
        }
    }

    // Record final run
    if current_bit == 1 {
        stats.ones_runs.push(current_run);
    } else {
        stats.zeros_runs.push(current_run);
    }

    stats
}

/// Run-length statistics.
#[derive(Debug, Default)]
pub struct RunLengthStats {
    pub ones_runs: Vec<usize>,
    pub zeros_runs: Vec<usize>,
}

impl RunLengthStats {
    /// Maximum run of ones.
    pub fn max_ones_run(&self) -> usize {
        self.ones_runs.iter().copied().max().unwrap_or(0)
    }

    /// Maximum run of zeros.
    pub fn max_zeros_run(&self) -> usize {
        self.zeros_runs.iter().copied().max().unwrap_or(0)
    }

    /// Average run length.
    pub fn average_run(&self) -> f64 {
        let total_runs = self.ones_runs.len() + self.zeros_runs.len();
        if total_runs == 0 {
            return 0.0;
        }

        let total_length: usize =
            self.ones_runs.iter().sum::<usize>() + self.zeros_runs.iter().sum::<usize>();

        total_length as f64 / total_runs as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prbs7_sequence_length() {
        let mut prbs = PrbsGenerator::new(7);
        let initial = prbs.state;

        // Generate full sequence
        for _ in 0..127 {
            prbs.next_bit();
        }

        // Should return to initial state
        assert_eq!(prbs.state, initial);
    }

    #[test]
    fn test_prbs_balance() {
        // PRBS should have nearly equal ones and zeros
        let mut prbs = PrbsGenerator::new(7);
        let bits = prbs.generate_bits(127);

        let ones: usize = bits.iter().map(|&b| b as usize).sum();
        let zeros = 127 - ones;

        // PRBS-7 should have 64 ones and 63 zeros (or vice versa)
        assert!((ones as i32 - zeros as i32).abs() <= 1);
    }

    #[test]
    fn test_waveform_generation() {
        let mut prbs = PrbsGenerator::new(7);
        let wf = prbs.generate_nrz(10, 64, Seconds::from_ps(1.0));

        assert_eq!(wf.samples.len(), 640);
        assert!(wf.samples.iter().all(|&v| v == 1.0 || v == -1.0));
    }
}

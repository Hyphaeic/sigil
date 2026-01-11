//! Physical units with type safety.
//!
//! These newtypes provide compile-time unit checking to prevent
//! mixing incompatible quantities (e.g., adding Hertz to Seconds).

use serde::{Deserialize, Serialize};
use std::ops::{Add, Div, Mul, Sub};

/// Time duration in seconds.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Default, Serialize, Deserialize)]
pub struct Seconds(pub f64);

impl Seconds {
    pub const ZERO: Self = Self(0.0);

    #[inline]
    pub fn from_ps(ps: f64) -> Self {
        Self(ps * 1e-12)
    }

    #[inline]
    pub fn from_ns(ns: f64) -> Self {
        Self(ns * 1e-9)
    }

    #[inline]
    pub fn from_us(us: f64) -> Self {
        Self(us * 1e-6)
    }

    #[inline]
    pub fn from_ms(ms: f64) -> Self {
        Self(ms * 1e-3)
    }

    #[inline]
    pub fn as_ps(&self) -> f64 {
        self.0 * 1e12
    }

    #[inline]
    pub fn as_ns(&self) -> f64 {
        self.0 * 1e9
    }

    /// Convert to frequency (reciprocal).
    #[inline]
    pub fn to_frequency(&self) -> Hertz {
        Hertz(1.0 / self.0)
    }
}

impl Add for Seconds {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl Sub for Seconds {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}

impl Mul<f64> for Seconds {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self(self.0 * rhs)
    }
}

impl Div<f64> for Seconds {
    type Output = Self;
    fn div(self, rhs: f64) -> Self {
        Self(self.0 / rhs)
    }
}

impl Div<Seconds> for Seconds {
    type Output = f64;
    fn div(self, rhs: Seconds) -> f64 {
        self.0 / rhs.0
    }
}

/// Frequency in Hertz.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Default, Serialize, Deserialize)]
pub struct Hertz(pub f64);

impl Hertz {
    pub const ZERO: Self = Self(0.0);

    #[inline]
    pub fn from_khz(khz: f64) -> Self {
        Self(khz * 1e3)
    }

    #[inline]
    pub fn from_mhz(mhz: f64) -> Self {
        Self(mhz * 1e6)
    }

    #[inline]
    pub fn from_ghz(ghz: f64) -> Self {
        Self(ghz * 1e9)
    }

    #[inline]
    pub fn as_ghz(&self) -> f64 {
        self.0 * 1e-9
    }

    #[inline]
    pub fn as_mhz(&self) -> f64 {
        self.0 * 1e-6
    }

    /// Convert to period (reciprocal).
    #[inline]
    pub fn to_period(&self) -> Seconds {
        Seconds(1.0 / self.0)
    }

    /// Angular frequency (omega = 2 * pi * f).
    #[inline]
    pub fn angular(&self) -> f64 {
        2.0 * std::f64::consts::PI * self.0
    }
}

impl Add for Hertz {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl Sub for Hertz {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}

impl Mul<f64> for Hertz {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self(self.0 * rhs)
    }
}

impl Div<f64> for Hertz {
    type Output = Self;
    fn div(self, rhs: f64) -> Self {
        Self(self.0 / rhs)
    }
}

/// Voltage in Volts.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Default, Serialize, Deserialize)]
pub struct Volts(pub f64);

impl Volts {
    pub const ZERO: Self = Self(0.0);

    #[inline]
    pub fn from_mv(mv: f64) -> Self {
        Self(mv * 1e-3)
    }

    #[inline]
    pub fn as_mv(&self) -> f64 {
        self.0 * 1e3
    }

    /// Convert to dBm assuming 50 ohm impedance.
    #[inline]
    pub fn to_dbm_50ohm(&self) -> f64 {
        let power_watts = self.0 * self.0 / 50.0;
        10.0 * (power_watts / 1e-3).log10()
    }
}

impl Add for Volts {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl Sub for Volts {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}

impl Mul<f64> for Volts {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self(self.0 * rhs)
    }
}

impl Div<f64> for Volts {
    type Output = Self;
    fn div(self, rhs: f64) -> Self {
        Self(self.0 / rhs)
    }
}

/// Impedance in Ohms.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Default, Serialize, Deserialize)]
pub struct Ohms(pub f64);

impl Ohms {
    /// Standard 50 ohm reference impedance.
    pub const Z0_50: Self = Self(50.0);

    /// Standard 75 ohm reference impedance.
    pub const Z0_75: Self = Self(75.0);

    /// Differential 100 ohm reference impedance.
    pub const Z0_DIFF_100: Self = Self(100.0);
}

impl Add for Ohms {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl Mul<f64> for Ohms {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self(self.0 * rhs)
    }
}

/// Data rate in bits per second.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Default, Serialize, Deserialize)]
pub struct BitsPerSecond(pub f64);

impl BitsPerSecond {
    #[inline]
    pub fn from_gbps(gbps: f64) -> Self {
        Self(gbps * 1e9)
    }

    #[inline]
    pub fn from_gtps(gtps: f64) -> Self {
        // GT/s (gigatransfers) = Gbps for NRZ
        Self(gtps * 1e9)
    }

    #[inline]
    pub fn as_gbps(&self) -> f64 {
        self.0 * 1e-9
    }

    /// Unit interval (bit period).
    #[inline]
    pub fn ui(&self) -> Seconds {
        Seconds(1.0 / self.0)
    }

    /// Fundamental frequency (half the data rate for NRZ).
    #[inline]
    pub fn nyquist(&self) -> Hertz {
        Hertz(self.0 / 2.0)
    }
}

/// PCIe generation specifications.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PcieGeneration {
    Gen1,  // 2.5 GT/s
    Gen2,  // 5.0 GT/s
    Gen3,  // 8.0 GT/s
    Gen4,  // 16.0 GT/s
    Gen5,  // 32.0 GT/s
    Gen6,  // 64.0 GT/s (PAM4)
}

impl PcieGeneration {
    /// Data rate for this generation.
    pub fn data_rate(&self) -> BitsPerSecond {
        match self {
            Self::Gen1 => BitsPerSecond::from_gtps(2.5),
            Self::Gen2 => BitsPerSecond::from_gtps(5.0),
            Self::Gen3 => BitsPerSecond::from_gtps(8.0),
            Self::Gen4 => BitsPerSecond::from_gtps(16.0),
            Self::Gen5 => BitsPerSecond::from_gtps(32.0),
            Self::Gen6 => BitsPerSecond::from_gtps(64.0),
        }
    }

    /// Unit interval for this generation.
    pub fn ui(&self) -> Seconds {
        self.data_rate().ui()
    }

    /// Whether this generation uses PAM4 signaling.
    pub fn is_pam4(&self) -> bool {
        matches!(self, Self::Gen6)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pcie_gen5_timing() {
        let gen5 = PcieGeneration::Gen5;
        let ui = gen5.ui();

        // 32 GT/s = 31.25 ps UI
        assert!((ui.as_ps() - 31.25).abs() < 0.01);
    }

    #[test]
    fn test_frequency_period_reciprocal() {
        let freq = Hertz::from_ghz(16.0);
        let period = freq.to_period();

        assert!((period.as_ps() - 62.5).abs() < 0.01);
        assert!((period.to_frequency().0 - freq.0).abs() < 1.0);
    }
}

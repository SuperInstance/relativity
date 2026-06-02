//! Time dilation and length contraction.

use crate::lorentz::gamma;
use crate::minkowski::C;
use serde::{Serialize, Deserialize};

/// Time dilation: moving clocks run slow.
/// Δt = γ · Δτ where Δτ is the proper time.
pub fn time_dilated(proper_time: f64, beta: f64) -> f64 {
    gamma(beta) * proper_time
}

/// Inverse: given observed Δt, find proper time Δτ = Δt / γ.
pub fn proper_time_from_dilated(dilated_time: f64, beta: f64) -> f64 {
    dilated_time / gamma(beta)
}

/// Length contraction: moving rulers are shorter.
/// L = L₀ / γ where L₀ is proper length.
pub fn length_contracted(proper_length: f64, beta: f64) -> f64 {
    proper_length / gamma(beta)
}

/// Inverse: given contracted length L, find proper length L₀ = γ·L.
pub fn proper_length_from_contracted(contracted_length: f64, beta: f64) -> f64 {
    gamma(beta) * contracted_length
}

/// Relativistic Doppler effect factor for source moving along line of sight.
/// Approaching: f_obs = f_src · √((1+β)/(1-β))
/// Receding: f_obs = f_src · √((1-β)/(1+β))
pub fn doppler_factor(beta: f64, approaching: bool) -> f64 {
    let sign = if approaching { 1.0 } else { -1.0 };
    ((1.0 + sign * beta) / (1.0 - sign * beta)).sqrt()
}

/// A relativistic reference frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceFrame {
    /// Velocity as fraction of c.
    pub beta: f64,
    /// Direction of motion.
    pub direction: [f64; 3],
}

impl ReferenceFrame {
    pub fn new(beta: f64, direction: [f64; 3]) -> Self {
        Self { beta, direction }
    }

    /// Rest frame.
    pub fn rest() -> Self {
        Self { beta: 0.0, direction: [1.0, 0.0, 0.0] }
    }

    /// Lorentz factor for this frame.
    pub fn gamma(&self) -> f64 {
        gamma(self.beta)
    }

    /// Time dilation factor.
    pub fn time_dilation_factor(&self) -> f64 {
        self.gamma()
    }

    /// Length contraction factor.
    pub fn length_contraction_factor(&self) -> f64 {
        1.0 / self.gamma()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_dilation_zero_velocity() {
        assert!((time_dilated(1.0, 0.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_time_dilation_half_c() {
        let g = gamma(0.5);
        assert!((time_dilated(1.0, 0.5) - g).abs() < 1e-10);
    }

    #[test]
    fn test_time_dilation_high_speed() {
        let dt = time_dilated(1.0, 0.99);
        assert!(dt > 5.0); // γ at 0.99c ≈ 7.09
    }

    #[test]
    fn test_proper_time_roundtrip() {
        let tau = 2.0;
        let beta = 0.6;
        let dt = time_dilated(tau, beta);
        let tau2 = proper_time_from_dilated(dt, beta);
        assert!((tau - tau2).abs() < 1e-10);
    }

    #[test]
    fn test_length_contraction_zero_velocity() {
        assert!((length_contracted(1.0, 0.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_length_contraction_half_c() {
        let g = gamma(0.5);
        assert!((length_contracted(10.0, 0.5) - 10.0 / g).abs() < 1e-10);
    }

    #[test]
    fn test_length_contraction_high_speed() {
        let l = length_contracted(100.0, 0.99);
        assert!(l < 20.0);
    }

    #[test]
    fn test_proper_length_roundtrip() {
        let l0 = 5.0;
        let beta = 0.8;
        let l = length_contracted(l0, beta);
        let l0_2 = proper_length_from_contracted(l, beta);
        assert!((l0 - l0_2).abs() < 1e-10);
    }

    #[test]
    fn test_doppler_approaching() {
        let f = doppler_factor(0.5, true);
        assert!(f > 1.0); // blueshift
    }

    #[test]
    fn test_doppler_receding() {
        let f = doppler_factor(0.5, false);
        assert!(f < 1.0); // redshift
    }

    #[test]
    fn test_doppler_reciprocal() {
        let app = doppler_factor(0.5, true);
        let rec = doppler_factor(0.5, false);
        assert!((app * rec - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_reference_frame_rest() {
        let frame = ReferenceFrame::rest();
        assert!((frame.gamma() - 1.0).abs() < 1e-10);
    }
}

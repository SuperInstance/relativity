//! Relativistic energy-momentum: E² = p²c² + m²c⁴, relativistic kinematics.

use crate::lorentz::gamma;
use crate::minkowski::C;
use serde::{Serialize, Deserialize};

/// Energy-momentum 4-vector (E/c, px, py, pz).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FourMomentum {
    pub e_over_c: f64,
    pub px: f64,
    pub py: f64,
    pub pz: f64,
}

impl FourMomentum {
    pub fn new(e_over_c: f64, px: f64, py: f64, pz: f64) -> Self {
        Self { e_over_c, px, py, pz }
    }

    /// Create from rest mass (kg) and velocity β = v/c.
    pub fn from_rest_mass_and_beta(rest_mass_kg: f64, beta: f64) -> Self {
        let g = gamma(beta);
        let p = g * rest_mass_kg * beta * C; // relativistic momentum
        let e = g * rest_mass_kg * C * C;     // total energy
        Self { e_over_c: e / C, px: p, py: 0.0, pz: 0.0 }
    }

    /// Total energy E.
    pub fn energy(&self) -> f64 {
        self.e_over_c * C
    }

    /// Rest energy: E₀ = mc².
    pub fn rest_energy(rest_mass_kg: f64) -> f64 {
        rest_mass_kg * C * C
    }

    /// Invariant mass squared (in natural units): (E/c)² - p².
    pub fn invariant_mass_squared(&self) -> f64 {
        let p2 = self.px * self.px + self.py * self.py + self.pz * self.pz;
        self.e_over_c * self.e_over_c - p2
    }

    /// Rest mass from invariant mass.
    pub fn rest_mass(&self) -> f64 {
        (self.invariant_mass_squared().max(0.0)).sqrt() / C
    }

    /// Momentum magnitude.
    pub fn momentum_magnitude(&self) -> f64 {
        (self.px * self.px + self.py * self.py + self.pz * self.pz).sqrt()
    }

    /// Kinetic energy: T = E - mc² = (γ - 1)mc².
    pub fn kinetic_energy(&self) -> f64 {
        self.energy() - Self::rest_energy(self.rest_mass())
    }

    /// Verify E² = (pc)² + (mc²)².
    pub fn verify_energy_momentum_relation(&self) -> bool {
        let e = self.energy();
        let pc = self.momentum_magnitude() * C;
        let m = self.rest_mass();
        let mc2 = m * C * C;
        (e * e - pc * pc - mc2 * mc2).abs() < 1e-3 * e * e.max(1.0)
    }
}

/// Relativistic kinetic energy: T = (γ - 1)mc².
pub fn relativistic_kinetic_energy(rest_mass: f64, beta: f64) -> f64 {
    (gamma(beta) - 1.0) * rest_mass * C * C
}

/// Total relativistic energy: E = γmc².
pub fn total_energy(rest_mass: f64, beta: f64) -> f64 {
    gamma(beta) * rest_mass * C * C
}

/// Relativistic momentum magnitude: p = γmv.
pub fn relativistic_momentum(rest_mass: f64, beta: f64) -> f64 {
    gamma(beta) * rest_mass * beta * C
}

/// Velocity from kinetic energy and rest mass.
pub fn beta_from_kinetic_energy(kinetic_energy: f64, rest_mass: f64) -> f64 {
    let ratio = kinetic_energy / (rest_mass * C * C);
    (1.0 - 1.0 / (1.0 + ratio).powi(2)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rest_energy() {
        let e = FourMomentum::rest_energy(1.0); // 1 kg
        assert!((e - C * C).abs() < 1e-3);
    }

    #[test]
    fn test_energy_momentum_relation_at_rest() {
        let p = FourMomentum::from_rest_mass_and_beta(1.0, 0.0);
        assert!(p.verify_energy_momentum_relation());
        assert!((p.momentum_magnitude()).abs() < 1e-10);
    }

    #[test]
    fn test_energy_momentum_relation_moving() {
        let p = FourMomentum::from_rest_mass_and_beta(1.0, 0.6);
        assert!(p.verify_energy_momentum_relation());
    }

    #[test]
    fn test_energy_momentum_relation_fast() {
        let p = FourMomentum::from_rest_mass_and_beta(1.0, 0.99);
        assert!(p.verify_energy_momentum_relation());
    }

    #[test]
    fn test_invariant_mass_equals_rest_mass() {
        let m0 = 2.0;
        let p = FourMomentum::from_rest_mass_and_beta(m0, 0.8);
        let m_inv = p.rest_mass();
        assert!((m_inv - m0).abs() / m0 < 1e-6);
    }

    #[test]
    fn test_kinetic_energy_at_rest_is_zero() {
        let ke = relativistic_kinetic_energy(1.0, 0.0);
        assert!(ke.abs() < 1e-10);
    }

    #[test]
    fn test_total_energy_at_rest() {
        let e = total_energy(1.0, 0.0);
        assert!((e - C * C).abs() < 1e-3);
    }

    #[test]
    fn test_momentum_zero_at_rest() {
        assert!((relativistic_momentum(1.0, 0.0)).abs() < 1e-10);
    }

    #[test]
    fn test_e_squared_formula() {
        let m = 1.0;
        let beta = 0.5;
        let e = total_energy(m, beta);
        let p = relativistic_momentum(m, beta);
        let pc = p * C;
        let mc2 = m * C * C;
        let lhs = e * e;
        let rhs = pc * pc + mc2 * mc2;
        assert!((lhs - rhs).abs() / lhs < 1e-6);
    }

    #[test]
    fn test_beta_from_kinetic_energy_roundtrip() {
        let beta = 0.6;
        let ke = relativistic_kinetic_energy(1.0, beta);
        let beta2 = beta_from_kinetic_energy(ke, 1.0);
        assert!((beta - beta2).abs() < 1e-6);
    }
}

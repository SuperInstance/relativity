//! Cosmological basics: Friedmann equations, Hubble parameter, scale factor.

use serde::{Serialize, Deserialize};

/// Gravitational constant (SI).
pub const G: f64 = 6.67430e-11;

/// Speed of light (SI).
pub const C: f64 = 299_792_458.0;

/// Friedmann equation: H² = (8πG/3)ρ - k/a² + Λ/3.
/// Returns the Hubble parameter H for given density, curvature, scale factor, and cosmological constant.
pub fn friedmann_hubble(density: f64, k: f64, a: f64, lambda: f64) -> f64 {
    let h_squared = 8.0 * std::f64::consts::PI * G * density / 3.0 - k / (a * a) + lambda / 3.0;
    if h_squared > 0.0 { h_squared.sqrt() } else { 0.0 }
}

/// First Friedmann equation (simplified): H² = H₀²[Ω_r/a⁴ + Ω_m/a³ + Ω_k/a² + Ω_Λ].
/// Returns H for given density parameters and scale factor.
pub fn friedmann_hubble_with_params(
    h0: f64,
    omega_r: f64,
    omega_m: f64,
    omega_k: f64,
    omega_lambda: f64,
    a: f64,
) -> f64 {
    let h_squared = h0 * h0 * (
        omega_r / a.powi(4)
        + omega_m / a.powi(3)
        + omega_k / a.powi(2)
        + omega_lambda
    );
    if h_squared > 0.0 { h_squared.sqrt() } else { 0.0 }
}

/// Second Friedmann equation (acceleration): ä/a = -(4πG/3)(ρ + 3p/c²) + Λ/3.
/// Returns ä/a (deceleration parameter analog).
pub fn friedmann_acceleration(density: f64, pressure: f64, lambda: f64) -> f64 {
    -4.0 * std::f64::consts::PI * G * (density + 3.0 * pressure / (C * C)) / 3.0 + lambda / 3.0
}

/// Critical density: ρ_c = 3H²/(8πG).
pub fn critical_density(h: f64) -> f64 {
    3.0 * h * h / (8.0 * std::f64::consts::PI * G)
}

/// Hubble distance: d_H = c / H₀.
pub fn hubble_distance(h0: f64) -> f64 {
    C / h0
}

/// Hubble time: t_H = 1 / H₀.
pub fn hubble_time(h0: f64) -> f64 {
    1.0 / h0
}

/// Cosmological redshift from scale factor: 1 + z = a_obs/a_emit = 1/a (if a_obs=1).
pub fn cosmological_redshift(a_emit: f64) -> f64 {
    1.0 / a_emit - 1.0
}

/// Scale factor from redshift: a = 1/(1+z).
pub fn scale_factor_from_redshift(z: f64) -> f64 {
    1.0 / (1.0 + z)
}

/// Luminosity distance (flat universe approximation): d_L = (1+z) * c/H₀ * ∫₀ᶻ dz'/E(z').
/// Simplified for matter-only: d_L ≈ (2c/H₀)(1+z - √(1+z)) for Ω_m=1, Ω_Λ=0.
pub fn luminosity_distance_matter_only(z: f64, h0: f64) -> f64 {
    (1.0 + z) * 2.0 * C / h0 * (1.0 - 1.0 / (1.0 + z).sqrt())
}

/// ΛCDM density parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LCDMParams {
    pub h0: f64,           // Hubble constant (s⁻¹)
    pub omega_m: f64,      // Matter density parameter
    pub omega_lambda: f64, // Dark energy density parameter
    pub omega_r: f64,      // Radiation density parameter
}

impl LCDMParams {
    /// Planck 2018 best-fit parameters.
    pub fn planck2018() -> Self {
        // H₀ ≈ 67.4 km/s/Mpc
        let h0 = 67.4e3 / 3.0857e22; // convert to s⁻¹
        Self {
            h0,
            omega_m: 0.315,
            omega_lambda: 0.685,
            omega_r: 9.1e-5,
        }
    }

    /// Curvature parameter Ω_k = 1 - Ω_m - Ω_Λ - Ω_r.
    pub fn omega_k(&self) -> f64 {
        1.0 - self.omega_m - self.omega_lambda - self.omega_r
    }

    /// Age of universe estimate (matter + Λ approximation).
    pub fn age_estimate(&self) -> f64 {
        // Simplified: t₀ ≈ 2/(3H₀√(Ω_Λ)) * asinh(√(Ω_Λ/Ω_m))
        let ratio = self.omega_lambda / self.omega_m;
        2.0 / (3.0 * self.h0 * self.omega_lambda.sqrt()) * ratio.sqrt().asinh()
    }

    /// Hubble parameter at scale factor a.
    pub fn h_at(&self, a: f64) -> f64 {
        friedmann_hubble_with_params(
            self.h0, self.omega_r, self.omega_m, self.omega_k(), self.omega_lambda, a,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_friedmann_flat_matter_only() {
        // For Ω_m=1, Ω_Λ=0: H² = H₀²/a³
        let h0 = 70.0e3 / 3.0857e22;
        let h = friedmann_hubble_with_params(h0, 0.0, 1.0, 0.0, 0.0, 1.0);
        assert!((h - h0).abs() / h0 < 1e-10);
    }

    #[test]
    fn test_friedmann_at_half_scale() {
        let h0 = 70.0e3 / 3.0857e22;
        let h = friedmann_hubble_with_params(h0, 0.0, 1.0, 0.0, 0.0, 0.5);
        // H = H₀ * a^(-3/2) = H₀ * (0.5)^(-3/2) = H₀ * 2√2
        let expected = h0 * 2.0_f64.sqrt() * 2.0;
        assert!((h - expected).abs() / expected < 1e-10);
    }

    #[test]
    fn test_critical_density() {
        let h0 = 70.0e3 / 3.0857e22;
        let rho_c = critical_density(h0);
        // Should be ~9.2 × 10⁻²⁷ kg/m³
        assert!(rho_c > 5e-27 && rho_c < 2e-26);
    }

    #[test]
    fn test_hubble_distance() {
        let h0 = 70.0e3 / 3.0857e22;
        let d = hubble_distance(h0);
        // c/H₀ ≈ 4.28 × 10²⁶ m
        assert!(d > 1e26 && d < 1e27);
    }

    #[test]
    fn test_cosmological_redshift() {
        assert!((cosmological_redshift(1.0) - 0.0).abs() < 1e-10);
        assert!((cosmological_redshift(0.5) - 1.0).abs() < 1e-10);
        assert!((cosmological_redshift(0.25) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_scale_factor_roundtrip() {
        let z = 2.0;
        let a = scale_factor_from_redshift(z);
        let z2 = cosmological_redshift(a);
        assert!((z - z2).abs() < 1e-10);
    }

    #[test]
    fn test_lcdm_planck_params() {
        let p = LCDMParams::planck2018();
        let omega_total = p.omega_m + p.omega_lambda + p.omega_r + p.omega_k();
        assert!((omega_total - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_lcdm_age_estimate() {
        let p = LCDMParams::planck2018();
        let age = p.age_estimate();
        // Should be ~13.8 Gyr ≈ 4.35 × 10¹⁷ s
        assert!(age > 3.5e17 && age < 5.0e17);
    }

    #[test]
    fn test_friedmann_acceleration_deceleration() {
        // Matter-only universe should decelerate (ä < 0)
        let acc = friedmann_acceleration(1e-26, 0.0, 0.0);
        assert!(acc < 0.0);
    }

    #[test]
    fn test_friedmann_acceleration_with_lambda() {
        // Large Λ should cause acceleration (ä > 0)
        let acc = friedmann_acceleration(1e-26, 0.0, 1e-35);
        assert!(acc > 0.0);
    }

    #[test]
    fn test_luminosity_distance_zero_redshift() {
        let h0 = 70.0e3 / 3.0857e22;
        let d = luminosity_distance_matter_only(0.0, h0);
        assert!(d.abs() < 1e-10);
    }
}

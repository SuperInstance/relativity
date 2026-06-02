//! Gravitational redshift.

use serde::{Serialize, Deserialize};

/// Gravitational redshift factor: λ_obs / λ_emit = √(1 - r_s/r_obs) / √(1 - r_s/r_emit).
/// For observer at infinity: λ_obs/λ_emit = 1/√(1 - r_s/r_emit).
pub fn gravitational_redshift_ratio(r_emit: f64, r_obs: f64, r_s: f64) -> f64 {
    let f_emit = (1.0 - r_s / r_emit).max(0.0).sqrt();
    let f_obs = (1.0 - r_s / r_obs).max(0.0).sqrt();
    f_obs / f_emit
}

/// Gravitational redshift z = Δλ/λ = (λ_obs - λ_emit)/λ_emit.
pub fn gravitational_redshift_z(r_emit: f64, r_obs: f64, r_s: f64) -> f64 {
    gravitational_redshift_ratio(r_emit, r_obs, r_s) - 1.0
}

/// Frequency ratio: f_obs / f_emit = 1 / (redshift ratio).
pub fn gravitational_frequency_ratio(r_emit: f64, r_obs: f64, r_s: f64) -> f64 {
    1.0 / gravitational_redshift_ratio(r_emit, r_obs, r_s)
}

/// Gravitational time dilation: dτ/dt = √(1 - r_s/r).
/// A clock at radius r ticks slower by this factor compared to infinity.
pub fn gravitational_time_dilation(r: f64, r_s: f64) -> f64 {
    (1.0 - r_s / r).max(0.0).sqrt()
}

/// Shapiro delay (approximate): extra time delay for light passing near a massive object.
/// Δt ≈ (4GM/c³) * ln(4·r_obs·r_emit / b²) where b is impact parameter.
pub fn shapiro_delay(r_emit: f64, r_obs: f64, impact_param: f64, mass_kg: f64) -> f64 {
    let g = 6.67430e-11;
    let c = 299_792_458.0;
    let factor = 4.0 * g * mass_kg / (c * c * c);
    let arg = 4.0 * r_obs * r_emit / (impact_param * impact_param);
    if arg > 0.0 {
        factor * arg.ln()
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_redshift_zero_at_same_radius() {
        let r = 1e8;
        let r_s = 3000.0;
        let z = gravitational_redshift_z(r, r, r_s);
        assert!(z.abs() < 1e-10);
    }

    #[test]
    fn test_redshift_positive_ascending() {
        // Light climbing out of a gravitational well is redshifted
        let r_s = 3000.0;
        let z = gravitational_redshift_z(1e6, 1e8, r_s);
        assert!(z > 0.0); // redshifted
    }

    #[test]
    fn test_redshift_approaches_infinity_near_horizon() {
        let r_s = 3000.0;
        let z = gravitational_redshift_z(r_s * 1.001, 1e10, r_s);
        assert!(z > 10.0); // extreme redshift
    }

    #[test]
    fn test_frequency_ratio_reciprocal() {
        let r_s = 3000.0;
        let r_emit = 1e6;
        let r_obs = 1e8;
        let lambda_ratio = gravitational_redshift_ratio(r_emit, r_obs, r_s);
        let freq_ratio = gravitational_frequency_ratio(r_emit, r_obs, r_s);
        assert!((lambda_ratio * freq_ratio - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_time_dilation_at_infinity() {
        let r_s = 3000.0;
        let r = 1e20; // very far
        let td = gravitational_time_dilation(r, r_s);
        assert!((td - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_time_dilation_at_r_s() {
        let r_s = 3000.0;
        let td = gravitational_time_dilation(r_s, r_s);
        assert!(td.abs() < 1e-10);
    }

    #[test]
    fn test_shapiro_delay_positive() {
        let delay = shapiro_delay(1.496e11, 1.496e11, 7e8, 1.989e30);
        assert!(delay > 0.0);
        // Shapiro delay for Sun is ~120 μs for a ray grazing the surface
        // Our parameters give a small but positive delay
    }

    #[test]
    fn test_redshift_earth_surface() {
        // Gravitational redshift from Earth's surface to infinity should be very small
        let r_s_earth = 2.0 * 6.67430e-11 * 5.972e24 / (299_792_458.0_f64).powi(2);
        let r_earth = 6.371e6;
        let z = gravitational_redshift_z(r_earth, 1e20, r_s_earth);
        assert!(z > 0.0);
        assert!(z < 1e-9); // very tiny
    }
}

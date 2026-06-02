//! Tensor formulation: metric tensor, Christoffel symbols for Schwarzschild metric.

use nalgebra::DMatrix;
use serde::{Serialize, Deserialize};

/// Schwarzschild radius: r_s = 2GM/c².
pub fn schwarzschild_radius(mass_kg: f64) -> f64 {
    let g = 6.67430e-11;
    let c = 299_792_458.0;
    2.0 * g * mass_kg / (c * c)
}

/// Schwarzschild metric components in Schwarzschild coordinates (t, r, θ, φ).
/// Returns the diagonal metric tensor as (g_tt, g_rr, g_θθ, g_φφ).
/// g_tt = -(1 - r_s/r), g_rr = (1 - r_s/r)^{-1}, g_θθ = r², g_φφ = r²sin²θ.
pub fn schwarzschild_metric(r: f64, theta: f64, r_s: f64) -> [f64; 4] {
    let f = 1.0 - r_s / r;
    let sin_theta = theta.sin();
    [
        -(f),                  // g_tt
        1.0 / f,               // g_rr
        r * r,                 // g_θθ
        r * r * sin_theta * sin_theta, // g_φφ
    ]
}

/// Compute Christoffel symbols Γ^μ_{νλ} for the Schwarzschild metric.
/// Returns a 4x4x4 array indexed as Γ[μ][ν][λ].
/// Only the non-zero components are computed.
pub fn schwarzschild_christoffel(r: f64, theta: f64, r_s: f64) -> [[[f64; 4]; 4]; 4] {
    let mut gamma = [[[0.0_f64; 4]; 4]; 4];
    let f = 1.0 - r_s / r;
    let r2 = r * r;
    let sin_th = theta.sin();
    let cos_th = theta.cos();
    let sin2 = sin_th * sin_th;

    // Γ^t_{tr} = Γ^t_{rt} = r_s / (2r²f)
    let val_tt_r = r_s / (2.0 * r2 * f);
    gamma[0][0][1] = val_tt_r;
    gamma[0][1][0] = val_tt_r;

    // Γ^r_{tt} = (c² r_s f) / (2r²) — we set c=1 in geometric units
    gamma[1][0][0] = r_s * f / (2.0 * r2);

    // Γ^r_{rr} = -r_s / (2r²f)
    gamma[1][1][1] = -r_s / (2.0 * r2 * f);

    // Γ^r_{θθ} = -(r - r_s)
    gamma[1][2][2] = -(r - r_s);

    // Γ^r_{φφ} = -(r - r_s) sin²θ
    gamma[1][3][3] = -(r - r_s) * sin2;

    // Γ^θ_{rθ} = Γ^θ_{θr} = 1/r
    gamma[2][1][2] = 1.0 / r;
    gamma[2][2][1] = 1.0 / r;

    // Γ^θ_{φφ} = -sinθ cosθ
    gamma[2][3][3] = -sin_th * cos_th;

    // Γ^φ_{rφ} = Γ^φ_{φr} = 1/r
    gamma[3][1][3] = 1.0 / r;
    gamma[3][3][1] = 1.0 / r;

    // Γ^φ_{θφ} = Γ^φ_{φθ} = cosθ/sinθ
    gamma[3][2][3] = cos_th / sin_th;
    gamma[3][3][2] = cos_th / sin_th;

    gamma
}

/// Metric tensor for FLRW cosmology: ds² = -c²dt² + a(t)²[dr²/(1-kr²) + r²dΩ²].
/// Returns simplified diagonal components (g_tt, g_rr, g_θθ, g_φφ).
pub fn flrw_metric(a: f64, k: f64, r: f64, theta: f64) -> [f64; 4] {
    let sin_th = theta.sin();
    [
        -1.0,                           // g_tt (with c=1)
        a * a / (1.0 - k * r * r),     // g_rr
        a * a * r * r,                  // g_θθ
        a * a * r * r * sin_th * sin_th, // g_φφ
    ]
}

/// A general 4x4 metric tensor wrapper.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricTensor {
    /// 4x4 matrix representation.
    pub components: [[f64; 4]; 4],
}

impl MetricTensor {
    pub fn new(components: [[f64; 4]; 4]) -> Self {
        Self { components }
    }

    /// Minkowski metric in spherical-like coordinates: diag(-1, 1, r², r²sin²θ).
    pub fn minkowski_spherical(r: f64, theta: f64) -> Self {
        let sin_th = theta.sin();
        Self::new([
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, r * r, 0.0],
            [0.0, 0.0, 0.0, r * r * sin_th * sin_th],
        ])
    }

    /// Get element g_μν.
    pub fn get(&self, mu: usize, nu: usize) -> f64 {
        self.components[mu][nu]
    }

    /// Inverse metric (only correct for diagonal metrics).
    pub fn inverse_diagonal(&self) -> Self {
        let mut inv = [[0.0; 4]; 4];
        for i in 0..4 {
            inv[i][i] = 1.0 / self.components[i][i];
        }
        Self::new(inv)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schwarzschild_radius_sun() {
        let r_s = schwarzschild_radius(1.989e30); // Sun
        assert!((r_s - 2953.0).abs() < 10.0); // ~2953 m
    }

    #[test]
    fn test_schwarzschild_radius_earth() {
        let r_s = schwarzschild_radius(5.972e24);
        // ~8.87 mm
        assert!(r_s > 0.008 && r_s < 0.01);
    }

    #[test]
    fn test_schwarzschild_metric_far_field() {
        let r_s = 3000.0;
        let metric = schwarzschild_metric(1e10, std::f64::consts::FRAC_PI_2, r_s);
        // Far from source, should approximate Minkowski
        assert!((metric[0] - (-1.0)).abs() < 1e-6);
        assert!((metric[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_schwarzschild_metric_at_horizon() {
        let r_s = 3000.0;
        let metric = schwarzschild_metric(r_s, std::f64::consts::FRAC_PI_2, r_s);
        assert!((metric[0]).abs() < 1e-6); // g_tt → 0
        assert!(metric[1] > 1e10); // g_rr → ∞
    }

    #[test]
    fn test_christoffel_symmetric_in_lower_indices() {
        let r_s = 3000.0;
        let r = 1e7;
        let theta = std::f64::consts::FRAC_PI_4;
        let gamma = schwarzschild_christoffel(r, theta, r_s);
        for mu in 0..4 {
            for nu in 0..4 {
                for lam in 0..4 {
                    assert!((gamma[mu][nu][lam] - gamma[mu][lam][nu]).abs() < 1e-10,
                        "Not symmetric: Γ^{}[{}][{}] != Γ^{}[{}][{}]",
                        mu, nu, lam, mu, lam, nu);
                }
            }
        }
    }

    #[test]
    fn test_christoffel_minkowski_limit() {
        // Far from any mass, Christoffel symbols should be small
        let r_s = 3000.0;
        let r = 1e15; // very far
        let theta = std::f64::consts::FRAC_PI_4;
        let gamma = schwarzschild_christoffel(r, theta, r_s);
        // Γ^t_{tr} should be tiny
        assert!(gamma[0][0][1].abs() < 1e-15);
    }

    #[test]
    fn test_flrw_metric_flat() {
        let m = flrw_metric(1.0, 0.0, 1.0, std::f64::consts::FRAC_PI_2);
        assert!((m[0] - (-1.0)).abs() < 1e-10);
        assert!((m[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_metric_tensor_inverse_diagonal() {
        let mt = MetricTensor::new([
            [-2.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 0.0],
            [0.0, 0.0, 0.0, 7.0],
        ]);
        let inv = mt.inverse_diagonal();
        assert!((inv.get(0, 0) - (-0.5)).abs() < 1e-10);
        assert!((inv.get(1, 1) - (1.0 / 3.0)).abs() < 1e-10);
    }
}

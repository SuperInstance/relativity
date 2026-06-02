//! Lorentz transformations: boosts, rotations, velocity addition.

use nalgebra::{Matrix4, Vector4};
use serde::{Serialize, Deserialize};
use crate::minkowski::{FourVector, C};

/// Lorentz factor γ = 1/√(1 - β²).
pub fn gamma(beta: f64) -> f64 {
    let beta2 = beta * beta;
    assert!(beta2 < 1.0, "β must be less than 1 (speed of light)");
    1.0 / (1.0 - beta2).sqrt()
}

/// Lorentz boost matrix along the x-axis.
pub fn boost_x(beta: f64) -> Matrix4<f64> {
    let g = gamma(beta);
    Matrix4::new(
        g, -g * beta, 0.0, 0.0,
        -g * beta, g, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    )
}

/// Lorentz boost matrix along the y-axis.
pub fn boost_y(beta: f64) -> Matrix4<f64> {
    let g = gamma(beta);
    Matrix4::new(
        g, 0.0, -g * beta, 0.0,
        0.0, 1.0, 0.0, 0.0,
        -g * beta, 0.0, g, 0.0,
        0.0, 0.0, 0.0, 1.0,
    )
}

/// Lorentz boost matrix along the z-axis.
pub fn boost_z(beta: f64) -> Matrix4<f64> {
    let g = gamma(beta);
    Matrix4::new(
        g, 0.0, 0.0, -g * beta,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        -g * beta, 0.0, 0.0, g,
    )
}

/// Apply a Lorentz boost to a 4-vector.
pub fn boost(v: &FourVector, beta: f64, axis: Axis) -> FourVector {
    let m = match axis {
        Axis::X => boost_x(beta),
        Axis::Y => boost_y(beta),
        Axis::Z => boost_z(beta),
    };
    let result = m * v.to_vector();
    FourVector::from_vector(&result)
}

/// Spatial rotation in the xy-plane.
pub fn rotation_xy(theta: f64) -> Matrix4<f64> {
    let c = theta.cos();
    let s = theta.sin();
    Matrix4::new(
        1.0, 0.0, 0.0, 0.0,
        0.0, c, -s, 0.0,
        0.0, s, c, 0.0,
        0.0, 0.0, 0.0, 1.0,
    )
}

/// Spatial rotation in the xz-plane.
pub fn rotation_xz(theta: f64) -> Matrix4<f64> {
    let c = theta.cos();
    let s = theta.sin();
    Matrix4::new(
        1.0, 0.0, 0.0, 0.0,
        0.0, c, 0.0, -s,
        0.0, 0.0, 1.0, 0.0,
        0.0, s, 0.0, c,
    )
}

/// Coordinate axis for boosts.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Axis {
    X, Y, Z,
}

/// Relativistic velocity addition: v_combined = (v1 + v2) / (1 + v1·v2/c²).
/// Returns β_combined as fraction of c.
pub fn velocity_addition(beta1: f64, beta2: f64) -> f64 {
    (beta1 + beta2) / (1.0 + beta1 * beta2)
}

/// Rapidity parameter: tanh(φ) = β.
pub fn rapidity(beta: f64) -> f64 {
    beta.atanh()
}

/// Inverse rapidity: β = tanh(φ).
pub fn from_rapidity(phi: f64) -> f64 {
    phi.tanh()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::minkowski::minkowski_metric;

    #[test]
    fn test_gamma_zero() {
        assert!((gamma(0.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_gamma_half() {
        let g = gamma(0.5);
        assert!((g - 1.0 / (0.75_f64).sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_gamma_near_c() {
        let g = gamma(0.9999);
        assert!(g > 50.0);
    }

    #[test]
    fn test_boost_x_identity() {
        let b = boost_x(0.0);
        let identity: Matrix4<f64> = Matrix4::identity();
        for i in 0..4 {
            for j in 0..4 {
                assert!((b[(i, j)] - identity[(i, j)]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_boost_x_preserves_interval() {
        let v = FourVector::new(10.0, 5.0, 3.0, 2.0);
        let s2 = v.interval();
        let b = boost_x(0.6);
        let v2_vec = b * v.to_vector();
        let v2 = FourVector::from_vector(&v2_vec);
        assert!((v2.interval() - s2).abs() < 1e-8);
    }

    #[test]
    fn test_boost_y_preserves_interval() {
        let v = FourVector::new(10.0, 5.0, 3.0, 2.0);
        let s2 = v.interval();
        let b = boost_y(0.8);
        let v2_vec = b * v.to_vector();
        let v2 = FourVector::from_vector(&v2_vec);
        assert!((v2.interval() - s2).abs() < 1e-8);
    }

    #[test]
    fn test_boost_z_preserves_interval() {
        let v = FourVector::new(10.0, 5.0, 3.0, 2.0);
        let s2 = v.interval();
        let b = boost_z(0.3);
        let v2_vec = b * v.to_vector();
        let v2 = FourVector::from_vector(&v2_vec);
        assert!((v2.interval() - s2).abs() < 1e-8);
    }

    #[test]
    fn test_lorentz_invariance_full() {
        // η = Λ^T η Λ should hold
        let eta = minkowski_metric();
        let lambda = boost_x(0.5);
        let product = lambda.transpose() * &eta * lambda;
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { eta[(i, j)] } else { 0.0 };
                assert!((product[(i, j)] - expected).abs() < 1e-8,
                    "Failed at ({}, {}): got {}, expected {}", i, j, product[(i, j)], expected);
            }
        }
    }

    #[test]
    fn test_velocity_addition_symmetric() {
        let r1 = velocity_addition(0.5, 0.3);
        let r2 = velocity_addition(0.3, 0.5);
        assert!((r1 - r2).abs() < 1e-10);
    }

    #[test]
    fn test_velocity_addition_less_than_c() {
        let r = velocity_addition(0.9, 0.9);
        assert!(r < 1.0);
        assert!((r - 0.994475).abs() < 0.001);
    }

    #[test]
    fn test_velocity_addition_zero() {
        assert!((velocity_addition(0.5, 0.0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_rapidity_roundtrip() {
        let beta = 0.6;
        let phi = rapidity(beta);
        let beta2 = from_rapidity(phi);
        assert!((beta - beta2).abs() < 1e-10);
    }

    #[test]
    fn test_rotation_preserves_interval() {
        let v = FourVector::new(10.0, 5.0, 3.0, 2.0);
        let s2 = v.interval();
        let r = rotation_xy(0.7);
        let v2 = FourVector::from_vector(&(r * v.to_vector()));
        assert!((v2.interval() - s2).abs() < 1e-8);
    }

    #[test]
    fn test_boost_function() {
        let v = FourVector::new(10.0, 5.0, 0.0, 0.0);
        let v2 = boost(&v, 0.0, Axis::X);
        assert!((v2.ct - v.ct).abs() < 1e-10);
        assert!((v2.x - v.x).abs() < 1e-10);
    }
}

//! Minkowski spacetime: 4-vectors, metric signature, spacetime intervals.

use nalgebra::{Vector4, Matrix4};
use serde::{Serialize, Deserialize};

/// Speed of light in m/s.
pub const C: f64 = 299_792_458.0;

/// Minkowski metric signature convention: (+, -, -, -).
/// Returns the metric tensor η_μν.
pub fn minkowski_metric() -> Matrix4<f64> {
    Matrix4::from_diagonal(&Vector4::new(1.0, -1.0, -1.0, -1.0))
}

/// A spacetime 4-vector (ct, x, y, z).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FourVector {
    pub ct: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl FourVector {
    pub fn new(ct: f64, x: f64, y: f64, z: f64) -> Self {
        Self { ct, x, y, z }
    }

    /// Create from spatial components and time (seconds).
    pub fn from_spatial(t: f64, x: f64, y: f64, z: f64) -> Self {
        Self { ct: C * t, x, y, z }
    }

    /// Convert to nalgebra Vector4 (ct, x, y, z).
    pub fn to_vector(&self) -> Vector4<f64> {
        Vector4::new(self.ct, self.x, self.y, self.z)
    }

    /// Create from nalgebra Vector4.
    pub fn from_vector(v: &Vector4<f64>) -> Self {
        Self { ct: v[0], x: v[1], y: v[2], z: v[3] }
    }

    /// Spacetime interval: s² = (ct)² - x² - y² - z².
    pub fn interval(&self) -> f64 {
        self.ct * self.ct - self.x * self.x - self.y * self.y - self.z * self.z
    }

    /// Whether this interval is timelike (s² > 0).
    pub fn is_timelike(&self) -> bool {
        self.interval() > 0.0
    }

    /// Whether this interval is spacelike (s² < 0).
    pub fn is_spacelike(&self) -> bool {
        self.interval() < 0.0
    }

    /// Whether this interval is lightlike/null (s² = 0).
    pub fn is_lightlike(&self) -> bool {
        self.interval().abs() < 1e-10
    }

    /// Proper time for a timelike interval: τ = √(s²) / c.
    pub fn proper_time(&self) -> f64 {
        self.interval().sqrt() / C
    }

    /// Contract with another 4-vector using Minkowski metric.
    pub fn dot(&self, other: &FourVector) -> f64 {
        self.ct * other.ct - self.x * other.x - self.y * other.y - self.z * other.z
    }

    /// Spatial distance: √(x² + y² + z²).
    pub fn spatial_norm(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Coordinate time in seconds.
    pub fn time(&self) -> f64 {
        self.ct / C
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minkowski_metric_diagonal() {
        let eta = minkowski_metric();
        assert_eq!(eta[(0, 0)], 1.0);
        assert_eq!(eta[(1, 1)], -1.0);
        assert_eq!(eta[(2, 2)], -1.0);
        assert_eq!(eta[(3, 3)], -1.0);
    }

    #[test]
    fn test_minkowski_metric_off_diagonal_zero() {
        let eta = minkowski_metric();
        assert_eq!(eta[(0, 1)], 0.0);
        assert_eq!(eta[(1, 2)], 0.0);
        assert_eq!(eta[(3, 0)], 0.0);
    }

    #[test]
    fn test_timelike_interval() {
        let v = FourVector::from_spatial(10.0, 0.0, 0.0, 0.0);
        assert!(v.is_timelike());
        assert!(v.interval() > 0.0);
    }

    #[test]
    fn test_spacelike_interval() {
        let v = FourVector::new(0.0, 1.0, 0.0, 0.0);
        assert!(v.is_spacelike());
        assert!(v.interval() < 0.0);
    }

    #[test]
    fn test_lightlike_interval() {
        let v = FourVector::new(1.0, 1.0, 0.0, 0.0);
        assert!(v.is_lightlike());
    }

    #[test]
    fn test_interval_value() {
        let v = FourVector::new(5.0, 3.0, 0.0, 0.0);
        assert!((v.interval() - 16.0).abs() < 1e-10);
    }

    #[test]
    fn test_four_vector_dot() {
        let a = FourVector::new(5.0, 3.0, 0.0, 0.0);
        let b = FourVector::new(4.0, 3.0, 0.0, 0.0);
        // 5*4 - 3*3 = 20 - 9 = 11
        assert!((a.dot(&b) - 11.0).abs() < 1e-10);
    }

    #[test]
    fn test_spatial_norm() {
        let v = FourVector::new(0.0, 3.0, 4.0, 0.0);
        assert!((v.spatial_norm() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_interval_invariant_under_lorentz_boost() {
        let v = FourVector::new(10.0, 6.0, 2.0, 1.0);
        let s2 = v.interval();
        // Apply a boost and check invariance
        let beta = 0.5;
        let s: f64 = 1.0 - beta * beta;
        let gamma: f64 = 1.0 / s.sqrt();
        let ct_prime = gamma * (v.ct - beta * v.x);
        let x_prime = gamma * (v.x - beta * v.ct);
        let v2 = FourVector::new(ct_prime, x_prime, v.y, v.z);
        assert!((v2.interval() - s2).abs() < 1e-10);
    }
}

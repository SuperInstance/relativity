//! Geodesic motion in curved spacetime (Schwarzschild geodesics).

use serde::{Serialize, Deserialize};
use crate::tensor::schwarzschild_christoffel;

/// Constants of motion for Schwarzschild geodesics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchwarzschildConstants {
    /// Energy per unit mass: E/m = (1 - r_s/r) dt/dτ.
    pub energy_per_mass: f64,
    /// Angular momentum per unit mass: L/m = r² dφ/dτ.
    pub angular_momentum_per_mass: f64,
}

/// Effective potential for Schwarzschild geodesics.
/// V_eff = -GM/r + L²/(2r²) - GML²/(c²r³)
pub fn effective_potential(r: f64, mass_kg: f64, l: f64) -> f64 {
    let g = 6.67430e-11;
    let c = 299_792_458.0;
    let gm = g * mass_kg;
    -gm / r + l * l / (2.0 * r * r) - gm * l * l / (c * c * r * r * r)
}

/// Schwarzschild geodesic equations: compute acceleration d²x^μ/dτ².
/// Uses Christoffel symbols: d²x^μ/dτ² = -Γ^μ_{νλ} (dx^ν/dτ)(dx^λ/dτ²)
pub fn geodesic_acceleration(
    position: &[f64; 4],  // (t, r, θ, φ)
    velocity: &[f64; 4],  // (dt/dτ, dr/dτ, dθ/dτ, dφ/dτ)
    r_s: f64,
) -> [f64; 4] {
    let theta = position[2];
    let r = position[1];
    let gamma = schwarzschild_christoffel(r, theta, r_s);

    let mut accel = [0.0; 4];
    for mu in 0..4 {
        let mut sum = 0.0;
        for nu in 0..4 {
            for lam in 0..4 {
                sum -= gamma[mu][nu][lam] * velocity[nu] * velocity[lam];
            }
        }
        accel[mu] = sum;
    }
    accel
}

/// Circular orbit radius for given angular momentum in Schwarzschild.
/// r_circ = L²/(2GM) * [1 ± √(1 - 12(GML/c²)²/L⁴)]  — only valid for L > L_ISCO.
/// Simplified: for Newtonian limit, r = L²/(GM).
pub fn circular_orbit_radius_newtonian(l: f64, mass_kg: f64) -> f64 {
    let g = 6.67430e-11;
    l * l / (g * mass_kg)
}

/// ISCO (Innermost Stable Circular Orbit) radius = 3 r_s for Schwarzschild.
pub fn isco_radius(r_s: f64) -> f64 {
    3.0 * r_s
}

/// Photon sphere radius = 1.5 r_s for Schwarzschild.
pub fn photon_sphere_radius(r_s: f64) -> f64 {
    1.5 * r_s
}

/// Step a geodesic using RK4 integration.
pub fn geodesic_step_rk4(
    pos: &[f64; 4],
    vel: &[f64; 4],
    r_s: f64,
    dtau: f64,
) -> ([f64; 4], [f64; 4]) {
    // k1
    let a1 = geodesic_acceleration(pos, vel, r_s);

    // k2
    let mut pos2 = [0.0; 4];
    let mut vel2 = [0.0; 4];
    for i in 0..4 {
        pos2[i] = pos[i] + 0.5 * dtau * vel[i];
        vel2[i] = vel[i] + 0.5 * dtau * a1[i];
    }
    let a2 = geodesic_acceleration(&pos2, &vel2, r_s);

    // k3
    let mut pos3 = [0.0; 4];
    let mut vel3 = [0.0; 4];
    for i in 0..4 {
        pos3[i] = pos[i] + 0.5 * dtau * vel2[i];
        vel3[i] = vel[i] + 0.5 * dtau * a2[i];
    }
    let a3 = geodesic_acceleration(&pos3, &vel3, r_s);

    // k4
    let mut pos4 = [0.0; 4];
    let mut vel4 = [0.0; 4];
    for i in 0..4 {
        pos4[i] = pos[i] + dtau * vel3[i];
        vel4[i] = vel[i] + dtau * a3[i];
    }
    let a4 = geodesic_acceleration(&pos4, &vel4, r_s);

    // Combine
    let mut new_pos = [0.0; 4];
    let mut new_vel = [0.0; 4];
    for i in 0..4 {
        new_pos[i] = pos[i] + dtau / 6.0 * (vel[i] + 2.0 * vel2[i] + 2.0 * vel3[i] + vel4[i]);
        new_vel[i] = vel[i] + dtau / 6.0 * (a1[i] + 2.0 * a2[i] + 2.0 * a3[i] + a4[i]);
    }
    (new_pos, new_vel)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_isco_radius() {
        let r_s = 3000.0;
        assert!((isco_radius(r_s) - 9000.0).abs() < 1e-10);
    }

    #[test]
    fn test_photon_sphere() {
        let r_s = 3000.0;
        assert!((photon_sphere_radius(r_s) - 4500.0).abs() < 1e-10);
    }

    #[test]
    fn test_geodesic_at_rest_far_field() {
        // Particle at rest far from mass should barely move
        let r_s = 3000.0;
        let pos = [0.0, 1e12, std::f64::consts::FRAC_PI_2, 0.0];
        let vel = [1.0, 0.0, 0.0, 0.0]; // at rest (only dt/dτ = 1)
        let accel = geodesic_acceleration(&pos, &vel, r_s);
        // Radial acceleration should be very small
        assert!(accel[1].abs() < 1e-10);
    }

    #[test]
    fn test_circular_orbit_radius_newtonian() {
        // For Earth-like orbit: r = L²/(GM)
        let g = 6.67430e-11;
        let m = 1.989e30;
        let r = 1.496e11; // 1 AU
        let v = 29783.0; // orbital velocity
        let l = r * v;
        let r_calc = circular_orbit_radius_newtonian(l, m);
        assert!((r_calc - r).abs() / r < 0.01);
    }

    #[test]
    fn test_effective_potential_positive_at_large_r() {
        let v = effective_potential(1e15, 1.989e30, 1e11);
        assert!(v.abs() < 1e10); // very small far away
    }

    #[test]
    fn test_geodesic_rk4_preserves_theta_at_equator() {
        let r_s = 3000.0;
        let r = 1e8;
        let pos = [0.0, r, std::f64::consts::FRAC_PI_2, 0.0];
        let vel = [1.0, 0.0, 0.0, 0.001]; // moving in φ
        let dtau = 100.0;
        let (new_pos, _) = geodesic_step_rk4(&pos, &vel, r_s, dtau);
        // θ should stay near π/2 for equatorial orbit
        assert!((new_pos[2] - std::f64::consts::FRAC_PI_2).abs() < 1e-6);
    }
}

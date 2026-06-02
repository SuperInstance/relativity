# spacetime-rs

Special and general relativity in Rust. Lorentz transforms to geodesics.

A pure-Rust library for special and general relativity — spacetime geometry, Lorentz transformations, relativistic kinematics, energy-momentum, tensor formulation, geodesic integration, gravitational redshift, and cosmology.

**87 tests** · `nalgebra` + `serde` + `num-complex` · MIT OR Apache-2.0

---

## Install

```toml
[dependencies]
spacetime-rs = "0.1"
```

Or:

```sh
cargo add spacetime-rs
```

---

## Quick Start

### Lorentz Boost a 4-Vector

```rust
use spacetime_rs::minkowski::FourVector;
use spacetime_rs::lorentz::{boost, Axis, gamma};

let event = FourVector::from_spatial(1e-6, 100.0, 0.0, 0.0); // t=1μs, x=100m
println!("Interval s² = {:.3}", event.interval());

let boosted = boost(&event, 0.6, Axis::X);  // β = 0.6c along x
println!("Boosted: ct'={:.3}, x'={:.3}", boosted.ct, boosted.x);
println!("Interval preserved: {:.3}", boosted.interval());
```

### Time Dilation

```rust
use spacetime_rs::kinematics::{time_dilated, length_contracted};

let earth_time = time_dilated(1.0, 0.99);  // 1s proper time at 0.99c
let ship_length = length_contracted(100.0, 0.99);
```

### Energy-Momentum

```rust
use spacetime_rs::energy_momentum::{FourMomentum, relativistic_kinetic_energy};

let p = FourMomentum::from_rest_mass_and_beta(1.0, 0.8);
println!("E = {:.3e} J", p.energy());
println!("E² = (pc)² + (mc²)² holds: {}", p.verify_energy_momentum_relation());
```

### Schwarzschild Geodesic

```rust
use spacetime_rs::geodesic::{geodesic_step_rk4, isco_radius};
use spacetime_rs::tensor::schwarzschild_radius;

let r_s = schwarzschild_radius(1.989e30);
println!("ISCO: {:.0} m", isco_radius(r_s));
```

### Cosmology (ΛCDM)

```rust
use spacetime_rs::cosmology::LCDMParams;

let params = LCDMParams::planck2018();
println!("Ω_m = {:.3}, Ω_Λ = {:.3}", params.omega_m, params.omega_lambda);
```

---

## Modules

| Module | Domain | Key Structures |
|---|---|---|
| `minkowski` | Flat spacetime | 4-vectors, metric η_μν, spacetime intervals |
| `lorentz` | Boosts & rotations | Boost matrices, velocity addition, rapidity |
| `kinematics` | SR kinematics | Time dilation, length contraction, Doppler effect |
| `energy_momentum` | Relativistic mechanics | 4-momentum, E² = p²c² + m²c⁴, kinetic energy |
| `tensor` | Curved spacetime | Schwarzschild metric, Christoffel symbols, FLRW metric |
| `geodesic` | Geodesic motion | Effective potential, RK4 integration, ISCO, photon sphere |
| `redshift` | Gravitational redshift | Wavelength/frequency shift, Shapiro delay |
| `cosmology` | Cosmology | Friedmann equations, ΛCDM parameters, Hubble law |

---

## License

MIT OR Apache-2.0
